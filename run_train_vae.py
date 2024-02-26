# credits:
# Flax code is adapted from https://github.com/huggingface/transformers/blob/main/examples/flax/vision/run_image_classification.py
# GAN related code are adapted from https://github.com/patil-suraj/vit-vqgan/
# Further adapted from https://github.com/cccntu/fine-tune-models/ by Jonathan Chang
import os

# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['TF_GPU_CUPTI_FORCE_CONCURRENT_KERNEL'] = '1'

from copy import deepcopy
from functools import partial
import gc

from pathlib import Path
from typing import Callable, Tuple, Union

import wandb
import numpy as np
import jax
# jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("./jax_cache")

from flax import traverse_util, linen as nn
from flax.serialization import to_bytes
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
import optax

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from diffusers.models.vae_flax import FlaxAutoencoderKL, FlaxDecoderOutput, FlaxAutoencoderKLOutput, FlaxDiagonalGaussianDistribution
from datasets import Dataset as HFDataset
from lpips_j.lpips import LPIPS

from utils.dataloaders import DecoderImageDataset, JaxBatchDataloader
from modeling.discriminator import NLayerDiscriminator, NLayerDiscriminatorConfig
from utils.train_states import TrainStateEma
from utils.loss_functions import srgb_to_oklab

USE_WANDB = True
PROFILE = False

TRAIN_EMA = False
EMA_DECAY = (1 - 0.001 / 6) # ~0.99983, EMA decay value used for sd-vae-ft adjusted for batch
# paths and configs
if USE_WANDB:
    wandb.init(project="vae-training")

# cards supporting bfloat16 can easily support batches of 8 for every 20GB or so
# recommend raising in increments of 8 until it OOMs then take a step back
BATCH_SIZE = 8
GRAD_ACC_STEPS = 1

# size of image inputs -- 256 is enough
SAMPLE_SIZE = 256

# learning rate, which will not be scaled by batch size
LEARNING_RATE = 5e-7

LOG_STEPS = 10 * GRAD_ACC_STEPS
EVAL_STEPS = 1000 * GRAD_ACC_STEPS
CHECKPOINT_STEPS = 25000 * GRAD_ACC_STEPS

# so multistep will respect latent batch splitting, and also so steps above aren't interfered with
GRAD_ACC_STEPS = GRAD_ACC_STEPS

WARMUP_STEPS = 5000 * GRAD_ACC_STEPS
TOTAL_STEPS = 150_000 * GRAD_ACC_STEPS

# You can skip the first N steps for discriminator training.  I leave it off.
DISC_LOSS_SKIP_STEPS = 0 * GRAD_ACC_STEPS

# will dump checkpoints every {CHECKPOINT_STEPS} steps to this directory, as FlaxAutoencoderKL checkpoints
CHECKPOINT_SAVE_PATH = "./checkpoints/"

# a huggingface dataset containing columns "path"
# path: can be absolute or relative to `DATA_ROOT`
DATA_ROOT = "/"
hfds = HFDataset.from_csv("../sd_vae_trainer/dataset.csv")

# this corresponds to a local dir containing the config.json file
# the config.json file is copied from https://github.com/patil-suraj/vit-vqgan/
DISC_CONFIG_PATH = "configs/vqgan/discriminator/config.json"

output_dir = Path("./output_vae")
output_dir.mkdir(exist_ok=True)

# TODO:
# KC loss trial run: Change NOTHING except adding kc loss, calibrating, and bumping disc loss up.
# General model run, on winning regime:
# - Set train duration to 300k steps
# - Set color space for reconstruction loss to LAB color space.
# - Taper off discriminator loss in proportion to other loss objective tapers.

# loss value weights
# reconstruction losses
LOSS_L1_ENABLED = True
COST_L1 = 1.0

LOSS_L2_ENABLED = False
COST_L2 = 0.0

LOSS_LPIPS_ENABLED = True
COST_LPIPS = 0.25

# WGAN-GP grad penalty
COST_GRAD_PENALTY = 1e1

LOSS_DISC_ENABLED = True
DISC_WEIGHT = 0.5

# Here lie dragons.
# Don't do this for finetuning, unless you know what you're doing and are prepared
# to re-train existing downstream models.
TRAIN_ENCODER = True

# KL regularization. CompVis used a small amount (1e-6).
LOSS_KL_ENABLED = True
COST_KL = 1e-6

# Setting these up as arrays and passing them in as arguments allows avoiding recompiles when tuning hyperparameters.
loss_weights = {
    "l1": jnp.array(COST_L1),
    "l2": jnp.array(COST_L2),
    "lpips": jnp.array(COST_LPIPS),
    "kl": jnp.array(COST_KL),
    "grad_penalty": jnp.array(COST_GRAD_PENALTY),
    "disc": jnp.array(DISC_WEIGHT),
}

# Which loss objectives are enabled.  Disabled ones won't be calculated.  Changing these will require recompiles.
loss_enabled = {
    "l1": LOSS_L1_ENABLED,
    "l2": LOSS_L2_ENABLED,
    "lpips": LOSS_LPIPS_ENABLED,
    "kl": LOSS_KL_ENABLED,
    "disc": LOSS_DISC_ENABLED,
}

# I highly recommend starting from "stabilityai/sd-vae-ft-mse" if using this for a SD1.5/2.1 decoder finetune.
# It is much better trained than the stock kl-f8 autoencoder from SD 1.5 and losses starting out will likely be lower.
vae, vae_params = FlaxAutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    from_pt=True,
    _do_init=True,
    dtype=jnp.bfloat16
)

# this is just to get pylance to behave
vae = vae # type: FlaxAutoencoderKL

# place our model on the accelerator
vae_params = jax.device_put(vae_params, jax.devices()[0])
loss_weights = jax.device_put(loss_weights, jax.devices()[0])

disc_model = NLayerDiscriminator(
    NLayerDiscriminatorConfig.from_pretrained("./disc_config.json"),
    seed=42,
    _do_init=True,
)

lpips_model = LPIPS()

def init_lpips(rng, image_size):
    x = jax.random.normal(rng, shape=(1, image_size, image_size, 3))
    return lpips_model.init(rng, x, x)

lpips_rng, training_rng, dataset_rng, valset_rng = jax.random.split(jax.random.PRNGKey(0), 4)

lpips_params = init_lpips(lpips_rng, image_size=SAMPLE_SIZE)

lr_schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(
            init_value=0.0,
            end_value=LEARNING_RATE,
            transition_steps=WARMUP_STEPS + 1,  # ensure not 0
        ),
        optax.constant_schedule(LEARNING_RATE)
    ],
    boundaries=[WARMUP_STEPS],
)

disc_loss_skip_schedule = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0),
        optax.constant_schedule(1),
    ],
    boundaries=[DISC_LOSS_SKIP_STEPS],
)

if TRAIN_ENCODER:
    param_partitions = traverse_util.path_aware_map(
        lambda path, v: 'trainable' if any(part in path for part in ["encoder", "quant_conv", "decoder", "post_quant_conv"]) else 'frozen', vae_params)
else:
    param_partitions = traverse_util.path_aware_map(
        lambda path, v: 'trainable' if any(part in path for part in ["decoder", "post_quant_conv"]) else 'frozen', vae_params)

optimizer = optax.multi_transform(
    {
        'trainable': optax.chain(
            optax.adamw(
                learning_rate=lr_schedule,
                b1=0.5,
                b2=0.9
            )
        ),
        'frozen': optax.set_to_zero()
    },
    param_partitions
)

optimizer_disc = optax.chain(
    optax.adamw(
        learning_rate=LEARNING_RATE,
        b1=0.5,
        b2=0.9
    )
)

optimizer = optax.MultiSteps(optimizer, GRAD_ACC_STEPS)
optimizer_disc = optax.MultiSteps(optimizer_disc, GRAD_ACC_STEPS)

# create training states
if TRAIN_EMA:
    train_state = TrainStateEma.create(
        apply_fn=vae.__call__,
        params=jax.device_put(vae_params),
        tx=optimizer,
        ema_decay=EMA_DECAY
    )
else:
    train_state = TrainState.create(
        apply_fn=vae.__call__,
        params=jax.device_put(vae_params),
        tx=optimizer
    )

train_state_disc = TrainState.create(
    apply_fn=disc_model,
    params=jax.device_put(disc_model.params),
    tx=optimizer_disc
)

train_state = jax.device_put(train_state, jax.devices()[0]) # type: TrainState
train_state_disc = jax.device_put(train_state_disc, jax.devices()[0]) # type: TrainState

def to_encoder(img: jax.Array) -> jax.Array:
    return jnp.transpose(img * 2.0 - 1.0, (0, 3, 1, 2))

def from_decoder(dec: jax.Array) -> jax.Array:
    return jnp.transpose(dec, (0, 2, 3, 1)) / 2 + 0.5


@jax.jit
def reconstruct(params: Union[dict, FrozenDict], original: jax.Array) -> jax.Array:
    decoder_out = vae.apply( # type: ignore
        {"params": params},
        to_encoder(original),
        sample_posterior=False
    ) # type: FlaxDecoderOutput
    return from_decoder(decoder_out.sample)

@jax.jit
def get_latent_dist(params: Union[dict, FrozenDict], original: jax.Array) -> Tuple[jax.Array, jax.Array]:
    latent_dist = vae.apply( # type: ignore
        {"params": params},
        to_encoder(original),
        return_dict=False,
        method=vae.encode
    )[0] # type: FlaxDiagonalGaussianDistribution
    return latent_dist.mean, latent_dist.logvar

@partial(jax.jit, donate_argnums=(0, 1), static_argnums=(5,))
def train_step(
    state: TrainState,
    train_rng: jax.Array,
    original: jax.Array,
    state_disc: TrainState,
    loss_weights: dict,
    loss_enabled: dict
) -> Tuple[TrainState, jax.Array, dict, jax.Array]:
    dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

    def encoder_loss(
        params
    ) -> Tuple[FlaxDiagonalGaussianDistribution, jax.Array]:

        # Compute latents given the current state of the encoder.
        current_latents = vae.apply( # type: ignore
            {"params": params},
            to_encoder(original),
            return_dict=False,
            method=vae.encode
        )[0] # type: FlaxDiagonalGaussianDistribution

        # Compute KL divergence loss for the latent space vs a standard gauissian.
        # This keeps the latent space locally smooth ("variational").
        loss_kl = jnp.mean(current_latents.kl()) if loss_enabled["kl"] else 0

        return current_latents, loss_kl

    def discriminator_loss(reconstruction):
        disc_fake_scores = state_disc.apply_fn(
            srgb_to_oklab(reconstruction),
            params=state_disc.params,
            dropout_rng=dropout_rng,
            train=False,
        )
        return jnp.mean(nn.softplus(-disc_fake_scores))

    def reconstruction_loss(
        original: jax.Array,
        reconstruction: jax.Array,
        simple: bool = False
    ) -> Tuple[jax.Array, dict]:
        lab_original = srgb_to_oklab(original)
        lab_reconstruction = srgb_to_oklab(reconstruction)

        loss_l1 = jnp.abs(lab_reconstruction - lab_original).mean() if loss_enabled["l1"] else 0
        loss_l2 = optax.l2_loss(lab_reconstruction, lab_original).mean() if loss_enabled["l2"] else 0
        loss_lpips = jnp.mean(lpips_model.apply(lpips_params, original, reconstruction)) if not simple and loss_enabled["lpips"] else 0

        loss_rec = (
            loss_l1 * loss_weights["l1"] +
            loss_l2 * loss_weights["l2"] +
            loss_lpips * loss_weights["lpips"]
        )

        loss_details = { "loss_rec": loss_rec }

        if loss_enabled["l1"]:
            loss_details['loss_mae'] = loss_l1

        if loss_enabled["l2"]:
            loss_details['loss_mse'] = loss_l2

        if not simple and loss_enabled["lpips"]:
            loss_details['loss_lpis'] = loss_lpips

        return loss_rec, loss_details

    def calculate_adaptive_weight(
        sample_rng: jax.Array
    ) -> jax.Array:

        def forward_over_last_layer(
            last_layer: jax.Array,
            params: dict,
            latent_dist: FlaxDiagonalGaussianDistribution,
            sample_rng: jax.Array
        ) -> jax.Array:
            # We need the whole params for the model but need the passed last layer to grad
            # Save the old last layer, we need it later, and replace it with the passed one
            old_lastlayer = params['decoder']['conv_out']['kernel']
            params['decoder']['conv_out']['kernel'] = last_layer

            decoder_out = vae.apply( # type: ignore
                {"params": params},
                latent_dist.sample(sample_rng),
                return_dict=False,
                method=vae.decode
            )[0] # type: FlaxDecoderOutput

            # Put the last layer back, so that this function is technically "side-effect free"
            params['decoder']['conv_out']['kernel'] = old_lastlayer

            return from_decoder(decoder_out)

        @jax.grad
        def compute_vae_loss_ll(
            last_layer: jax.Array,
            params: dict,
            latent: jax.Array,
            original: jax.Array,
            sample_rng: jax.Array
        ) -> jax.Array:
            reconstruction = forward_over_last_layer(last_layer, params, latent, sample_rng)

            loss_rec, _ = reconstruction_loss(original, reconstruction)

            return loss_rec

        @jax.grad
        def compute_disc_loss_ll(
            last_layer: jax.Array,
            params: dict,
            latent: jax.Array,
            sample_rng: jax.Array
        ) -> jax.Array:
            reconstruction = forward_over_last_layer(last_layer, params, latent, sample_rng)
            return discriminator_loss(reconstruction)

        latent_dist = vae.apply( # type: ignore
            {"params": state.params},
            to_encoder(original),
            return_dict=False,
            method=vae.encode
        )[0] # type: FlaxDiagonalGaussianDistribution

        rec_grads = compute_vae_loss_ll(
            state.params['decoder']['conv_out']['kernel'],
            state.params,
            latent_dist,
            original,
            sample_rng
        )
        disc_grads = compute_disc_loss_ll(
            state.params['decoder']['conv_out']['kernel'],
            state.params,
            latent_dist,
            sample_rng
        )

        # Calculate the adaptive weight
        d_weight = jnp.linalg.norm(rec_grads) / (jnp.linalg.norm(disc_grads) + 1e-4)
        d_weight = jnp.clip(d_weight, 0.0, 1e4)
        d_weight = d_weight * loss_weights["disc"]

        return jax.lax.stop_gradient(d_weight)

    @partial(jax.grad, has_aux=True)
    def compute_loss(
        params: dict,
        d_weight,
        latent_dist: Union[FlaxDiagonalGaussianDistribution, None],
        original: jax.Array,
        sample_rng: jax.Array
    ):
        latent_dist, loss_kl = encoder_loss(params, latent_dist)

        decoder_out = vae.apply( # type: ignore
            {"params": params},
            latent_dist.sample(sample_rng),
            return_dict=False,
            method=vae.decode
        )[0] # type: FlaxDecoderOutput
        reconstruction = from_decoder(decoder_out)

        loss_disc = discriminator_loss(reconstruction) if loss_enabled["disc"] else 0.0
        loss_rec, loss_details = reconstruction_loss(original, reconstruction)

        loss = (
            loss_rec +
            loss_kl * loss_weights["kl"] +
            loss_disc * d_weight
        )

        loss_details['loss_obj'] = loss
        loss_details['loss_disc'] = loss_disc
        loss_details['d_weight'] = d_weight
        loss_details['learning_rate'] = lr_schedule(state.step)

        if TRAIN_ENCODER:
            loss_details['loss_kl'] = loss_kl

        return loss, (loss_details, reconstruction)

    d_weight = calculate_adaptive_weight(sample_rng) if loss_enabled["disc"] else 0.0
    grad, (loss_details, reconstruction) = compute_loss(
        state.params,
        d_weight,
        original,
        sample_rng
    )

    new_state = state.apply_gradients(grads=grad)
    return new_state, new_train_rng, loss_details, reconstruction

@partial(jax.jit, donate_argnums=(0, 1))
def train_step_disc(
    state_disc: TrainState,
    train_rng: jax.Array,
    original: jax.Array,
    reconstruction: jax.Array,
    loss_weights: dict
) -> Tuple[TrainState, jax.Array, dict]:

    @partial(jax.grad, has_aux=True)
    def compute_stylegan_loss(
        disc_params: dict,
        real_images: jax.Array,
        fake_images: jax.Array,
        dropout_rng: jax.Array,
        disc_model_fn: Callable[..., jax.Array]
    ) -> Tuple[jax.Array, dict]:

        # Forward pass for both real and fake images
        disc_real_scores = disc_model_fn(
            real_images,
            params=disc_params,
            dropout_rng=dropout_rng,
            train=True
        )  # type: jax.Array
        disc_fake_scores = disc_model_fn(
            fake_images,
            params=disc_params,
            dropout_rng=dropout_rng,
            train=True
        )  # type: jax.Array

        # -log sigmoid(f(x)) = log (1 + exp(-f(x))) = softplus(-f(x))
        # -log(1-sigmoid(f(x))) = log (1 + exp(f(x))) = softplus(f(x))
        # https://github.com/pfnet-research/sngan_projection/issues/18#issuecomment-392683263
        loss_real = nn.softplus(-disc_real_scores) # type: jax.Array
        loss_fake = nn.softplus(disc_fake_scores) # type: jax.Array
        disc_loss_stylegan = jnp.mean(loss_real + loss_fake)

        # loss_real = jnp.mean(jax.nn.relu(1. - disc_real_scores))
        # loss_fake = jnp.mean(jax.nn.relu(1. + disc_fake_scores))
        # disc_loss_stylegan = 0.5 * (loss_real + loss_fake)

        # gradient penalty r1: https://github.com/NVlabs/stylegan2/blob/bf0fe0baba9fc7039eae0cac575c1778be1ce3e3/training/loss.py#L63-L67
        r1_grads = jax.grad(
            lambda x: jnp.mean(
                disc_model_fn(
                    x,
                    params=disc_params,
                    dropout_rng=dropout_rng,
                    train=False)
            )
        )(real_images)
        # get the squares of gradients
        r1_grads = jnp.mean(r1_grads**2)

        disc_loss = disc_loss_stylegan + loss_weights["grad_penalty"] * r1_grads
        disc_loss_details = {
            "pred_p_real": jnp.exp(-loss_real).mean(),  # p = 1 -> predict real is real
            "pred_p_fake": jnp.exp(-loss_fake).mean(),  # p = 1 -> predict fake is fake
            "loss_real": loss_real.mean(),
            "loss_fake": loss_fake.mean(),
            "loss_stylegan": disc_loss_stylegan,
            "loss_gradient_penalty": loss_weights["grad_penalty"] * r1_grads,
            "loss": disc_loss,
        }
        return disc_loss, disc_loss_details

    dropout_rng, new_train_rng = jax.random.split(train_rng)

    # convert fake images to int then back to float, so discriminator can't cheat
    dtype = reconstruction.dtype
    reconstruction = (reconstruction.clip(0, 1) * 255).astype(jnp.uint8).astype(dtype) / 255

    disc_grads, disc_loss_details = compute_stylegan_loss(
        state_disc.params,
        srgb_to_oklab(original),
        srgb_to_oklab(reconstruction),
        dropout_rng,
        disc_model,
    )

    disc_loss_details = disc_loss_details | {"learning_rate_disc": LEARNING_RATE } # lr_schedule(state_disc.step)}
    state_disc = state_disc.apply_gradients(grads=disc_grads)

    # metrics = jax.lax.pmean(metrics, axis_name="batch")
    return state_disc, new_train_rng, disc_loss_details

# data loader without shuffle, so we can see the progress on the same images
# Take the first 128 images as validation set
train_ds = DecoderImageDataset(hfds.select(range(128, len(hfds))), SAMPLE_SIZE, root=DATA_ROOT) # type: ignore
test_ds = DecoderImageDataset(hfds.select(range(128)), SAMPLE_SIZE, root=DATA_ROOT) # type: ignore

dataloader = JaxBatchDataloader(dataset_rng, BATCH_SIZE, train_ds)
test_dl = JaxBatchDataloader(valset_rng, BATCH_SIZE, test_ds, only_once=True)

if USE_WANDB:
    wandb.log({"train_dataset_size": len(train_ds)})

# evaluation functions

def infer_fn(batch: dict, state: TrainState) -> jax.Array:
    return reconstruct(state.params, batch["original"])

def infer_fn_get_latent_dist(batch: dict, state: TrainState) -> FlaxDiagonalGaussianDistribution:
    mean, logvar = get_latent_dist(state.params, batch["original"])
    return FlaxDiagonalGaussianDistribution(jnp.concatenate([mean, logvar], axis=-1))

def infer_fn_ema(batch: dict, state: TrainStateEma) -> jax.Array:
    return reconstruct(state.ema_params, batch["original"])

eval_batches = []
def evaluate(use_tqdm=False, step=None) -> None:
    losses = []
    losses_ema = []
    if len(eval_batches) == 0:
        iterable = test_dl if not use_tqdm else tqdm(test_dl)
        for batch in iterable:
            eval_batches.append(batch)
            if len(eval_batches) >= 128//BATCH_SIZE:
                break

    for batch in eval_batches:
        reconstruction = infer_fn(batch, train_state)
        losses.append(optax.l2_loss(reconstruction, batch["original"]).mean())
        if TRAIN_EMA:
            reconstruction_ema = infer_fn_ema(batch, train_state)
            losses_ema.append(optax.l2_loss(reconstruction_ema, batch["original"]).mean())
    loss = np.mean(jax.device_get(losses))
    if TRAIN_EMA:
        loss_ema = np.mean(jax.device_get(losses_ema))
    if USE_WANDB:
        wandb.log({"test_loss": loss}, step=step)
        if TRAIN_EMA:
            wandb.log({"test_loss_ema": loss_ema}, step=step)
    print(f"done eval")

def postpro(decoded_images: np.ndarray) -> list:
    """util function to postprocess images"""
    if np.any(np.isnan(decoded_images)):
        print("CRITICAL: decoded images contain NaN!")
    decoded_images = decoded_images.clip(0.0, 1.0)

    return [
        Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        for decoded_img in decoded_images
    ]

def log_chart(array, name, step):
    square_latent = np.concatenate([array[:, :, 0:2], array[:, :, 2:4]], axis=0)
    square_latent = np.concatenate([square_latent[:, :, 0], square_latent[:, :, 1]], axis=1)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=256)
    im = ax.imshow(square_latent, cmap='plasma')
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    wandb.log({name: wandb.Image(fig)}, step=step)
    plt.close()

def log_images(batches, num_images=8, suffix="", step=None) -> None:
    logged_images = 0

    def batch_gen():
        while True:
            for batch in batches:
                yield batch

    batch_iter = batch_gen()
    while logged_images < num_images:
        batch = next(batch_iter) # type: dict

        names = batch["name"]
        reconstruction = infer_fn(batch, train_state)
        if TRAIN_ENCODER:
            current_latent_dist = infer_fn_get_latent_dist(batch, train_state)
            current_logvars = np.split(current_latent_dist.logvar.astype(jnp.float32), current_latent_dist.logvar.shape[0], axis=0)
            current_means = np.split(current_latent_dist.mean.astype(jnp.float32), current_latent_dist.mean.shape[0], axis=0)

        if TRAIN_EMA:
            reconstruction_ema = infer_fn_ema(batch, train_state)
            left_right = np.concatenate([batch["original"], reconstruction, reconstruction_ema], axis=2)
        else:
            left_right = np.concatenate([batch["original"], reconstruction], axis=2)

        images = postpro(left_right)
        if USE_WANDB:
            for idx, (name, image) in enumerate(zip(names, images)):
                wandb.log(
                    {f"{name}{suffix}": wandb.Image(image, caption=name)}, step=step
                )
                if TRAIN_ENCODER:
                    log_chart(
                        current_logvars[idx][0],
                        f"Latent Space/{name}-logvar-current",
                        step
                    )
                    log_chart(
                        current_means[idx][0],
                        f"Latent Space/{name}-mean-current",
                        step
                    )
        logged_images += len(images)

def log_test_images(num_images=8, step=None) -> None:
    log_images(eval_batches, num_images=num_images, step=step)

def save_checkpoint(state: Union[TrainState, TrainStateEma]):
    if USE_WANDB:
        vae.save_pretrained(f"{CHECKPOINT_SAVE_PATH}{wandb.run.name}_step{state.step}/", state.params) # type: ignore
        if TRAIN_EMA:
            vae.save_pretrained(f"{CHECKPOINT_SAVE_PATH}{wandb.run.name}_step{state.step}-ema/", state.ema_params)

def data_iter():
    while True:
        for batch in dataloader:
            yield batch

@jax.jit
def encode_latent_for_cache(original: jax.Array):
    posterior = vae.apply( # type: ignore
        {"params": train_state.params},
        to_encoder(original),
        method=vae.encode
    ) # type: FlaxAutoencoderKLOutput
    batch = {
        "original": jnp.clip(jnp.round(original * 255), 0, 255).astype(jnp.uint8),
        "latent_dist": posterior.latent_dist
    }
    return batch

@jax.jit
def vae_decode_only(latent: jax.Array):
    decoder_out = vae.apply( # type: ignore
        {"params": vae_params},
        latent,
        method=vae.decode
    ) # type: FlaxDecoderOutput

    return from_decoder(decoder_out.sample)

metrics_dict = {}
metrics_list = []
steps_since_log = 0

for steps, batch in tqdm(enumerate(dataloader), total=TOTAL_STEPS, desc="Training...", dynamic_ncols=True):
    if PROFILE and steps == 100:
        jax.profiler.start_trace("./tensorboard")
    batch["original"] = jax.device_put(batch["original"], jax.devices()[0])

    train_state, training_rng, metrics, fake = train_step(
        train_state,
        training_rng,
        batch["original"],
        train_state_disc,
        loss_weights,
        loss_enabled
    )

    if disc_loss_skip_schedule(train_state.step) > 0 and loss_enabled["disc"]:
        train_state_disc, training_rng, metrics["disc_step"] = train_step_disc(
            train_state_disc,
            training_rng,
            batch["original"],
            fake,
            loss_weights,
        )
    else:
        metrics["disc_step"] = {}

    if PROFILE and steps == 110:
        print(metrics)
        jax.profiler.stop_trace()

    steps_since_log += 1
    try:
        metrics_dict = jax.tree_map(lambda x, y: x + y, metrics, metrics_dict)
    except ValueError:
        zero_tree = jax.tree_map(lambda x: 0.0, metrics) # type: dict
        zero_tree.update(metrics_dict)
        metrics_dict = jax.tree_map(lambda x, y: x + y, metrics, zero_tree)

    if steps % LOG_STEPS == 1:
        metrics_dict = jax.tree_map(lambda x: x / steps_since_log, metrics_dict)
        if USE_WANDB:
            wandb.log(metrics_dict, step=steps)
        metrics_dict = jax.tree_map(lambda x: 0.0, metrics_dict)
        steps_since_log = 0

    if steps % EVAL_STEPS == 1:
        evaluate(step=steps)
        log_test_images(step=steps)
        with Path(output_dir / "latest_state_disc.msgpack").open("wb") as f:
            f.write(to_bytes(jax.device_get(train_state_disc)))

        with Path(output_dir / "latest_state.msgpack").open("wb") as f:
            f.write(to_bytes(jax.device_get(train_state)))

        gc.collect()

    if steps % CHECKPOINT_STEPS == 1:
        save_checkpoint(train_state)

    if steps == TOTAL_STEPS:
        break

save_checkpoint(train_state)