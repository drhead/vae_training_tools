# credits:
# Flax code is adapted from https://github.com/huggingface/transformers/blob/main/examples/flax/vision/run_image_classification.py
# GAN related code are adapted from https://github.com/patil-suraj/vit-vqgan/
# Further adapted from https://github.com/cccntu/fine-tune-models/ by Jonathan Chang
import os
from queue import Queue
from threading import Thread
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "true"
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
os.environ['TF_GPU_CUPTI_FORCE_CONCURRENT_KERNEL'] = '1'
import random
from copy import deepcopy
from functools import partial
import gc
import math
from pathlib import Path
from typing import Callable, Tuple, Union, List, Sequence, Any

import wandb
import numpy as np
import jax
# jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc
cc.initialize_cache("./jax_cache")
import flax
from flax import traverse_util, linen as nn
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.serialization import to_bytes
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict
import optax
import torch
torch.manual_seed(0)
import torchvision.transforms as T
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from io import BytesIO
from PIL import Image, ImageCms
Image.MAX_IMAGE_PIXELS = None
from diffusers.models.vae_flax import FlaxAutoencoderKL, FlaxDecoderOutput, FlaxAutoencoderKLOutput
from datasets import Dataset as HFDataset
from lpips_j.lpips import LPIPS
import jaxwt as jwt

USE_WANDB = True
PROFILE = False

TRAIN_EMA = True
EMA_DECAY = (1 - 0.001 / 6) # ~0.99983, EMA decay value used for sd-vae-ft adjusted for batch
# paths and configs
if USE_WANDB:
    wandb.init(project="fluffy-vae")

# cards supporting bfloat16 can easily support batches of 16 for every 20GB or so
# recommend raising in increments of 8 until it OOMs then take a step back
BATCH_SIZE = 32
GRAD_ACC_STEPS = 1

# Split latent cache batches into N fragments.
LATENT_BATCH_SPLIT = 1
# original learning rate for VAE from LDM repo
LEARNING_RATE = 5e-7 # 1.0e-6 * BATCH_SIZE * GRAD_ACC_STEPS

LOG_STEPS = 10 * GRAD_ACC_STEPS
EVAL_STEPS = 1000 * GRAD_ACC_STEPS
CHECKPOINT_STEPS = 25000 * GRAD_ACC_STEPS

# so multistep will respect latent batch splitting, and also so steps above aren't interfered with
GRAD_ACC_STEPS = GRAD_ACC_STEPS * LATENT_BATCH_SPLIT

WARMUP_STEPS = 5000 * GRAD_ACC_STEPS
TOTAL_STEPS = 150_000 * GRAD_ACC_STEPS
# skip disc loss for the first 5000 steps, because discriminator is not trained yet
DISC_LOSS_SKIP_STEPS = 0 * GRAD_ACC_STEPS
L1_L2_SWITCH_STEPS = 100000 * GRAD_ACC_STEPS


# will dump checkpoints every {CHECKPOINT_STEPS} steps to this directory, as FlaxAutoencoderKL checkpoints
CHECKPOINT_SAVE_PATH = "/mnt/foxhole/checkpoints/sd_vae_trainer/"

# a huggingface dataset containing columns "path"
# path: can be absolute or relative to `DATA_ROOT`
DATA_ROOT = "/"
hfds = HFDataset.from_csv("./dataset.csv")

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
# reconstruction losses -- current weights match LDM/SD-VAE-FT values
COST_L1 = 1.0
COST_L2 = 10.0
COST_LPIPS = 1.0

# kurtosis concentration loss, for more natural images
COST_KC = 0.0 # -- try 0.0003 for first KC run and calibrate to be equal to L1/L2.

# WGAN-GP grad penalty
COST_GRAD_PENALTY = 1e1
DISC_WEIGHT = 0.5

# recursive reconstruction loss weight. expensive! should in theory reduce overall noise
RRC_WEIGHT = 0.0

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

# make a copy of the original VAE so we can compare its outputs to our trained model periodically
original_params = deepcopy(vae_params)
# don't forget to place it on the accelerator
original_params = jax.device_put(original_params, jax.devices()[0])

class RandomResizedSquareCrop(T.RandomResizedCrop):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(1.0, 1.0), interpolation=TF.InterpolationMode.LANCZOS):
        super().__init__(size, scale=scale, ratio=ratio, interpolation=interpolation)

    def get_params(self, img: torch.Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        _, height, width = TF.get_dimensions(img)
        area = height * width

        for _ in range(10):
            scale_min, scale_max = scale
            size_x, size_y = self.size
            scale_min = max(size_x * size_y / area, scale_min)
            target_area = random.uniform(scale_min, scale_max) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop if the max attempts are reached
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = int(round(height * min(ratio)))
            h = height
        elif in_ratio > max(ratio):
            h = int(round(width / max(ratio)))
            w = width
        else:
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2

        return i, j, h, w

class RandomDownscale(torch.nn.Module):
    def __init__(self, sizes=[640, 720, 960, 1080], side="longest", interpolation=TF.InterpolationMode.LANCZOS):
        super().__init__()

        self.sizes = sizes
        self.side = side
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, height, width = TF.get_dimensions(img)

        match self.side:
            case "width":
                scale_dim = width
            case "height":
                scale_dim = height
            case "shortest":
                scale_dim = min(width, height)
            case "longest":
                scale_dim = max(width, height)

        sizes = [size for size in self.sizes if size >= scale_dim]
        if len(sizes) == 0:
            return img

        scale = scale_dim / random.choice(sizes)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        return TF.resize(img, [new_height, new_width], interpolation=self.interpolation)

class RandomRotate90(T.RandomRotation):
    def __call__(self, img: torch.Tensor):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle)

class DecoderImageDataset(torch.utils.data.Dataset):
    def __init__(self, hfds, root=None):
        """hdfs: HFDataset"""
        self.hfds = hfds
        self.root = root

    def __len__(self):
        return len(self.hfds)

    def __getitem__(self, idx):
        attempt = 0
        while True:
            try:
                example = self.hfds[idx]
                # indices = example["indices"]
                path = example["path"]
                if self.root is not None:
                    path = os.path.join(self.root, path.lstrip("/"))
                orig_arr = self.load(path)
                return {
                    # "indices": indices,
                    "original": orig_arr,
                    "name": Path(path).name,
                }
            except:
                idx = random.randint(0, len(self.hfds) - 1)
                attempt += 1
                if attempt > 10:
                    print(f"Error reading image at index {idx}: {attempt} attempts done, either you're VERY unlucky or the dataloader broke")

    @staticmethod
    def load(path):
        img = Image.open(path)

        # ensure image is in sRGB color space
        icc_raw = img.info.get('icc_profile')
        if icc_raw:
            img = ImageCms.profileToProfile(
                img,
                ImageCms.ImageCmsProfile(BytesIO(icc_raw)),
                ImageCms.createProfile(colorSpace='sRGB'),
                outputMode='RGB'
            )
        elif img.mode != "RGB":
            img = img.convert("RGB")

        transform = T.Compose([
            # Random cropping of a square region with size at least 256x256
            RandomResizedSquareCrop(
                size=(256, 256),
                scale=(0.08, 1.0),
                ratio=(1.0, 1.0),
                interpolation=TF.InterpolationMode.LANCZOS
                ),
            # Random rotation in 90 degree increments
            RandomRotate90(0),
            # Random horizontal and vertical flips
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            # Convert to tensor
            T.ToTensor()
        ])

        # For downscaling to a fixed set of resolutions, followed by cropping:
        # RandomDownscale(),
        # RandomCrop(size=(256, 256),

        # Apply transformations
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img.permute(0, 2, 3, 1).numpy()

    @staticmethod
    def collate_fn(examples, return_names=False):
        res = {
            # "indices": [example["indices"] for example in examples],
            "original": np.concatenate(
                [example["original"] for example in examples], axis=0
            ),
        }
        if return_names:
            res["name"] = [example["name"] for example in examples] # type: ignore
        return res

class DiscriminatorBlock(nn.Module):
    features: int
    kernel_size: Sequence[int]
    strides: Union[int, Sequence[int]]
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=1,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False
        )
        self.norm = nn.LayerNorm()
        # self.norm = nn.BatchNorm(
        #     momentum=0.9
        # )


    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        x = self.conv(x)
        x = self.norm(x) #, use_running_average=not training)
        x = nn.leaky_relu(x, 0.2)
        return x

class NLayerDiscriminatorConfig(PretrainedConfig):
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = (256, 256),
        ndf: int = 64,
        n_layers: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.ndf = ndf
        self.n_layers = n_layers

class NLayerDiscriminatorModule(nn.Module):
    config: NLayerDiscriminatorConfig
    
    def setup(self):
        self.ndf = self.config.ndf
        self.n_layers = self.config.n_layers
        kw = (4, 4)
        padw = 1
        self.first_conv = nn.Conv(
            features=self.ndf,
            kernel_size=kw,
            strides=2,
            padding=padw
        )
        blocks = []
        nf_mult = 1
        for n in range(1, self.n_layers):
            nf_mult = min(2 ** n, 8)
            blocks.append(DiscriminatorBlock(
                features=self.ndf * nf_mult,
                kernel_size=kw,
                strides=2
            ))
        nf_mult = min(2 ** self.n_layers, 8)
        blocks.append(DiscriminatorBlock(
            features=self.ndf * nf_mult,
            kernel_size=kw,
            strides=1
        ))
        self.blocks = blocks
        self.last_conv = nn.Conv(
            features=1,
            kernel_size=kw,
            strides=1
        )

    def __call__(self, x: jnp.ndarray, training: bool = False):
        x = self.first_conv(x)
        x = nn.leaky_relu(x, 0.2)
        for block in self.blocks:
            x = block(x, training)
        x = self.last_conv(x)
        return x

class NLayerDiscriminatorPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = NLayerDiscriminatorConfig
    base_model_prefix = "model"
    module_class: nn.Module = None

    def __init__(
        self,
        config: NLayerDiscriminatorConfig,
        input_shape=(1, 256, 256, 3),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        module = self.module_class(config=config, **kwargs)
        super().__init__(
            config,
            module,
            input_shape=input_shape,
            seed=seed,
            dtype=dtype,
            _do_init=_do_init,
        )

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple) -> FrozenDict:
        # init input tensors
        pixel_values = jnp.zeros(input_shape, dtype=jnp.float32)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}
        variables = self.module.init(rngs, pixel_values, training=False)
        return variables["params"]

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
    ):
        # Handle any PRNG if needed
        rngs = {"dropout": dropout_rng} if dropout_rng is not None else {}
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values),
            train,
            rngs=rngs
        )

class NLayerDiscriminator(NLayerDiscriminatorPreTrainedModel):
    module_class = NLayerDiscriminatorModule

disc_model = NLayerDiscriminator(
    NLayerDiscriminatorConfig.from_pretrained("./disc_config.json"),
    seed=42,
    _do_init=True,
)

lpips_model = LPIPS()

def init_lpips(rng, image_size):
    x = jax.random.normal(rng, shape=(1, image_size, image_size, 3))
    return lpips_model.init(rng, x, x)

lpips_rng, training_rng = jax.random.split(jax.random.PRNGKey(0))

lpips_params = init_lpips(lpips_rng, image_size=256)

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


# Cosine down from 1 to 0 over the course of the run, for switching from L1 to L2 gradually
l1_loss_schedule = optax.cosine_decay_schedule(1.0, L1_L2_SWITCH_STEPS, 0.6)

disc_loss_skip_schedule = optax.join_schedules(
    schedules=[
        optax.constant_schedule(0),
        optax.constant_schedule(1),
    ],
    boundaries=[DISC_LOSS_SKIP_STEPS],
)

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
    optax.adamw(learning_rate=LEARNING_RATE,
                b1=0.5,
                b2=0.9
            )
)

optimizer = optax.MultiSteps(optimizer, GRAD_ACC_STEPS)
optimizer_disc = optax.MultiSteps(optimizer_disc, GRAD_ACC_STEPS)

class TrainStateEma(TrainState):
    ema_params: flax.core.FrozenDict[str, Any] = flax.struct.field(pytree_node=True)
    ema_decay: float = 0.999

    @classmethod
    def create(cls, *, apply_fn, params, tx, ema_decay, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt_state = tx.init(params_with_opt)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=deepcopy(params),
            ema_decay=ema_decay,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
        grads: Gradients that have the same pytree structure as `.params`.
        **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
        An updated instance of `self` with `step` incremented by one, `params`
        and `opt_state` updated by applying `grads`, and additional attributes
        replaced as specified by `kwargs`.
        """
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                'params': new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT]
            }
        else:
            new_params = new_params_with_opt

        new_ema_params = jax.tree_map(
            lambda ema, param: self.ema_decay * ema + (1.0 - self.ema_decay) * param,
            self.ema_params,
            new_params
        )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )

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

@jax.jit
def reconstruct(params: Union[dict, FrozenDict], original: jax.Array) -> jax.Array:
    decoder_out = vae.apply( # type: ignore
        {"params": params},
        jnp.transpose(original * 2.0 - 1.0, (0, 3, 1, 2)),
        sample_posterior=False,
        deterministic=True) # type: FlaxDecoderOutput
    return jnp.transpose(decoder_out.sample, (0, 2, 3, 1)) / 2 + 0.5

def kurtosis(x: jax.Array, axis=None) -> jax.Array:
    epsilon = 1e-10
    mean_squared = jnp.mean(x ** 2, axis=axis)
    mean_squared = jnp.where(mean_squared < epsilon, mean_squared + epsilon, mean_squared)
    kurt = (jnp.mean(x ** 4, axis=axis) / mean_squared ** 2) - 3
    return kurt

def real_softmax(x: jax.Array, axis=None) -> jax.Array:
    return jax.scipy.special.logsumexp(x, axis=axis)

def real_softmin(x: jax.Array, axis=None) -> jax.Array:
    return -jax.scipy.special.logsumexp(-x, axis=axis)

def kurtosis_concentration(x: jax.Array) -> jax.Array:
    coeffs_kt = jnp.zeros((20, x.shape[0]))
    # NOTE: Due to shape mismatches this can't be turned into lax
    for i in range(1,28):
        cA, (cH, cV, cD) = jwt.wavedec2(x, f"db{i}", level=1, mode="reflect")

        kurt = kurtosis(jnp.stack([cA, cH, cV, cD], 1), axis=(1, 2, 3)) # type: ignore
        coeffs_kt = coeffs_kt.at[i].set(kurt)

    return jnp.mean(real_softmax(coeffs_kt, axis=0) - real_softmin(coeffs_kt, axis=0))

def compute_kc_loss_grey(rgb: jax.Array) -> jax.Array:
    rgb = jnp.transpose(rgb, (0, 3, 1, 2))
    # convert image to grayscale
    weights = jnp.array([0.299, 0.587, 0.114])
    # Perform the weighted sum along the channel axis
    grey = jnp.sum(rgb * weights[None, :, None, None], axis=1, keepdims=True)

    return kurtosis_concentration(grey)

def compute_kc_loss_lab(lab: jax.Array) -> jax.Array:
    lab = jnp.transpose(lab, (0, 3, 1, 2))
    return kurtosis_concentration(lab[:, 0, :, :])

def srgb_to_oklab(srgb: jax.Array) -> jax.Array:
    # Convert to linear RGB space.
    rgb = jnp.where(
        srgb <= 0.04045,
        srgb / 12.92,
        # clipping avoids NaNs in backwards pass
        ((jnp.maximum(srgb, 0.04045) + 0.055) / 1.055) ** 2.4
    )

    # Convert RGB to LMS (cone response)
    t_rgb_lms = jnp.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])

    lin_lms = jnp.tensordot(rgb, t_rgb_lms, axes=[3, 0])

    # Cone response cuts off at low light and we rely on rods, but assume a
    # linear response in low light to preserve differentiablity.
    # (2/255) / 12.92, which is roughly in the range of scotopic vision
    # (2e-6 cd/m^2) given a bright 800x600 CRT at 250 cd/m^2.

    X = 6e-4
    A = jnp.cbrt(X) / X

    lms = jnp.where(
        lin_lms <= X,
        lin_lms * A,
        # clipping avoids NaNs in backwards pass
        jnp.cbrt(jnp.maximum(lin_lms, 6e-4))
    )

    # Convert LMS to Oklab
    t_lms_oklab = jnp.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])

    return jnp.tensordot(lms, t_lms_oklab, axes=[3, 0])

@partial(jax.jit, donate_argnums=(0, 1))
def train_step(
        state: TrainState,
        train_rng: jax.Array,
        original: jax.Array,
        state_disc: TrainState
) -> Tuple[TrainState, jax.Array, dict, jax.Array]:

    dropout_rng, new_train_rng = jax.random.split(train_rng)

    def calculate_adaptive_weight(rec_grads, disc_grads):
        # Calculate the adaptive weight
        d_weight = jnp.linalg.norm(rec_grads) / (jnp.linalg.norm(disc_grads) + 1e-4)
        d_weight = jnp.clip(d_weight, 0.0, 1.0)
        d_weight = d_weight * DISC_WEIGHT
        return jax.lax.stop_gradient(d_weight)

    def forward_over_last_layer(
            last_layer: jax.Array,
            params: dict,
            original: jax.Array,
            sample_rng: jax.Array
        ) -> jax.Array:
        # We need the whole params for the model but need the passed last layer to grad
        # Save the old last layer, we need it later, and replace it with the passed one
        old_lastlayer = params['decoder']['conv_out']['kernel']
        params['decoder']['conv_out']['kernel'] = last_layer
        decoder_out = vae.apply( # type: ignore
            {"params": params},
            jnp.transpose(original * 2.0 - 1.0, (0, 3, 1, 2)),
            sample_posterior=True,
            deterministic=False,
            rngs={"gaussian": sample_rng}) # type: FlaxDecoderOutput
        # Put the last layer back, so that this function is technically "side-effect free"
        params['decoder']['conv_out']['kernel'] = old_lastlayer
        return jnp.transpose(decoder_out.sample, (0, 2, 3, 1)) / 2 + 0.5

    @jax.grad
    def compute_rec_loss(
            last_layer: jax.Array,
            params: dict,
            original: jax.Array,
            sample_rng: jax.Array
        ) -> jax.Array:
        reconstruction = forward_over_last_layer(last_layer, params, original, sample_rng)

        # Convert the image to Lab color space so loss prioritizes human perception of color
        lab_original = srgb_to_oklab(original)
        lab_reconstruction = srgb_to_oklab(reconstruction)
        loss_lpips = jnp.mean(lpips_model.apply(lpips_params, original, reconstruction)) # type: ignore
        loss_l1 = jnp.abs(lab_reconstruction - lab_original).mean()
        loss_l2 = optax.l2_loss(lab_reconstruction, lab_original).mean()
        loss_kc = compute_kc_loss_lab(lab_reconstruction) if COST_KC > 0 else 0
        loss_rec = (
            loss_kc * COST_KC +
            loss_l1 * COST_L1 * l1_loss_schedule(state.step) +
            loss_l2 * COST_L2 * (1.0 - l1_loss_schedule(state.step)) + # type: ignore
            loss_lpips * COST_LPIPS
        )

        return loss_rec

    @jax.grad
    def compute_disc_loss(last_layer: jax.Array, params: dict, original: jax.Array, sample_rng: jax.Array, dropout_rng: jax.Array):
        reconstruction = forward_over_last_layer(last_layer, params, original, sample_rng)

        disc_fake_scores = state_disc.apply_fn(
            srgb_to_oklab(reconstruction),
            params=state_disc.params,
            dropout_rng=dropout_rng,
            train=False,
        )
        # return -jnp.mean(disc_fake_scores)
        return jnp.mean(nn.softplus(-disc_fake_scores))

    @partial(jax.grad, has_aux=True)
    def compute_loss(params: dict, d_weight, original: jax.Array, sample_rng: jax.Array, dropout_rng: jax.Array):
        decoder_out = vae.apply( # type: ignore
            {"params": params},
            jnp.transpose(original * 2.0 - 1.0, (0, 3, 1, 2)),
            sample_posterior=True,
            deterministic=False,
            rngs={"gaussian": sample_rng}) # type: FlaxDecoderOutput
        reconstruction = jnp.transpose(decoder_out.sample, (0, 2, 3, 1)) / 2 + 0.5

        disc_fake_scores = state_disc.apply_fn(
            srgb_to_oklab(reconstruction),
            params=state_disc.params,
            dropout_rng=dropout_rng,
            train=False,
        )
        lab_original = srgb_to_oklab(original)
        lab_reconstruction = srgb_to_oklab(reconstruction)
        loss_disc = jnp.mean(nn.softplus(-disc_fake_scores))
        # loss_disc = -jnp.mean(disc_fake_scores)
        loss_lpips = jnp.mean(lpips_model.apply(lpips_params, original, reconstruction)) # type: ignore
        loss_l1 = jnp.abs(lab_reconstruction - lab_original).mean()
        loss_l2 = optax.l2_loss(lab_reconstruction, lab_original).mean()
        loss_kc = compute_kc_loss_lab(lab_reconstruction) if COST_KC > 0 else 0
        loss_rec = (
            loss_kc * COST_KC +
            loss_l1 * COST_L1 * l1_loss_schedule(state.step) +
            loss_l2 * COST_L2 * (1.0 - l1_loss_schedule(state.step)) + # type: ignore
            loss_lpips * COST_LPIPS
        )

        loss = loss_rec + loss_disc * d_weight * disc_loss_skip_schedule(state.step)
        loss_details = {
            "loss_kc": loss_kc * COST_KC,
            "loss_mae": loss_l1 * COST_L1,
            "loss_l2": loss_l2 * COST_L2,
            "loss_lpips": loss_lpips * COST_LPIPS,
            "loss_disc": loss_disc, # * d_weight,
            "loss_obj": loss,
            "loss_rec": loss_rec,
            "d_weight": d_weight
        }
        return loss, (loss_details, reconstruction)

    sample_rng, dropout_rng = jax.random.split(dropout_rng, 2)
    # sample_rng is reused, this is intentional
    rec_grad = compute_rec_loss(
        state.params['decoder']['conv_out']['kernel'],
        state.params,
        original,
        sample_rng
    )
    disc_grad = compute_disc_loss(
        state.params['decoder']['conv_out']['kernel'],
        state.params,
        original,
        sample_rng,
        dropout_rng
    )
    grad, (loss_details, reconstruction) = compute_loss(
        state.params,
        calculate_adaptive_weight(rec_grad, disc_grad),
        original,
        sample_rng,
        dropout_rng
    )
    # legacy code, I didn't use multi gpu
    # grad = jax.lax.pmean(grad, "batch")
    loss_details = loss_details | {"learning_rate": lr_schedule(state.step)} # type: dict
    new_state = state.apply_gradients(grads=grad)

    # metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, new_train_rng, loss_details, reconstruction

@jax.profiler.annotate_function
@partial(jax.jit, donate_argnums=(0, 1))
def train_step_lc(
        state: TrainState,
        train_rng: jax.Array,
        original: jax.Array,
        latent: jax.Array,
        state_disc: TrainState
) -> Tuple[TrainState, jax.Array, dict, jax.Array]:

    dropout_rng, new_train_rng = jax.random.split(train_rng)

    @jax.profiler.annotate_function
    def discriminator_loss(reconstruction):
        disc_fake_scores = state_disc.apply_fn(
            srgb_to_oklab(reconstruction),
            params=state_disc.params,
            dropout_rng=dropout_rng,
            train=False,
        )
        return jnp.mean(nn.softplus(-disc_fake_scores))
    
    @jax.profiler.annotate_function
    def reconstruction_loss(original, reconstruction):
        lab_original = srgb_to_oklab(original)
        lab_reconstruction = srgb_to_oklab(reconstruction)
        loss_lpips = jnp.mean(lpips_model.apply(lpips_params, original, reconstruction)) # type: ignore
        loss_l1 = jnp.abs(lab_reconstruction - lab_original).mean()
        loss_l2 = optax.l2_loss(lab_reconstruction, lab_original).mean()
        loss_kc = compute_kc_loss_lab(lab_reconstruction) if COST_KC > 0 else 0
        loss_rec = (
            loss_kc * COST_KC +
            loss_l1 * COST_L1 * l1_loss_schedule(state.step) +
            loss_l2 * COST_L2 * (1.0 - l1_loss_schedule(state.step)) + # type: ignore
            loss_lpips * COST_LPIPS
        )
        loss_details = {
            "loss_kc": loss_kc * COST_KC,
            "loss_mae": loss_l1 * COST_L1,
            "loss_l2": loss_l2 * COST_L2,
            "loss_lpips": loss_lpips * COST_LPIPS,
            "loss_rec": loss_rec
        }
        return loss_rec, loss_details

    # Recursive Reconstruction Consistency loss function.
    # Intended to help maintain alignment between the encoder and decoder.
    @jax.profiler.annotate_function
    def recursive_consistency_loss(params, reconstruction):
        decoder_out = vae.apply( # type: ignore
            {"params": params},
            jnp.transpose(reconstruction * 2.0 - 1.0, (0, 3, 1, 2)),
            sample_posterior=False,
            deterministic=True) # type: FlaxDecoderOutput
        reconstruction_prime = jnp.transpose(decoder_out.sample, (0, 2, 3, 1)) / 2 + 0.5
        return reconstruction_loss(reconstruction, reconstruction_prime)

    @jax.profiler.annotate_function
    def calculate_adaptive_weight():
        @jax.profiler.annotate_function
        def forward_over_last_layer(
                last_layer: jax.Array,
                params: dict,
                latent: jax.Array
            ) -> jax.Array:
            # We need the whole params for the model but need the passed last layer to grad
            # Save the old last layer, we need it later, and replace it with the passed one
            old_lastlayer = params['decoder']['conv_out']['kernel']
            params['decoder']['conv_out']['kernel'] = last_layer
            decoder_out = vae.apply( # type: ignore
                {"params": params},
                latent,
                deterministic=False,
                method=vae.decode) # type: FlaxDecoderOutput
            # Put the last layer back, so that this function is technically "side-effect free"
            params['decoder']['conv_out']['kernel'] = old_lastlayer
            return jnp.transpose(decoder_out.sample, (0, 2, 3, 1)) / 2 + 0.5

        @jax.profiler.annotate_function
        @jax.grad
        def compute_rec_loss_ll(
                last_layer: jax.Array,
                params: dict,
                latent: jax.Array,
                original: jax.Array
            ) -> jax.Array:
            reconstruction = forward_over_last_layer(last_layer, params, latent)

            loss_rec, _ = reconstruction_loss(original, reconstruction)
            loss_rrc, _ = recursive_consistency_loss(params, reconstruction) if RRC_WEIGHT > 0 else (0, {})

            return loss_rec + loss_rrc * RRC_WEIGHT
        
        @jax.profiler.annotate_function
        @jax.grad
        def compute_disc_loss_ll(
                last_layer: jax.Array,
                params: dict,
                latent: jax.Array
            ) -> jax.Array:
            reconstruction = forward_over_last_layer(last_layer, params, latent)

            return discriminator_loss(reconstruction)
        
        rec_grads = compute_rec_loss_ll(
            state.params['decoder']['conv_out']['kernel'],
            state.params,
            latent,
            original
        )
        disc_grads = compute_disc_loss_ll(
            state.params['decoder']['conv_out']['kernel'],
            state.params,
            latent
        )
        # Calculate the adaptive weight
        d_weight = jnp.linalg.norm(rec_grads) / (jnp.linalg.norm(disc_grads) + 1e-4)
        d_weight = jnp.clip(d_weight, 0.0, 1e4)
        d_weight = d_weight * DISC_WEIGHT
        return jax.lax.stop_gradient(d_weight)

    @jax.profiler.annotate_function
    @partial(jax.grad, has_aux=True)
    def compute_loss(
            params: dict,
            d_weight,
            latent: jax.Array,
            original: jax.Array
        ):

        decoder_out = vae.apply( # type: ignore
            {"params": params},
            latent,
            deterministic=False,
            method=vae.decode) # type: FlaxDecoderOutput
        reconstruction = jnp.transpose(decoder_out.sample, (0, 2, 3, 1)) / 2 + 0.5

        loss_disc = discriminator_loss(reconstruction)
        loss_rec, loss_details = reconstruction_loss(original, reconstruction)
        loss_rrc, rrc_loss_details = recursive_consistency_loss(params, reconstruction) if RRC_WEIGHT > 0 else (0, {})

        loss = loss_rec + loss_rrc * RRC_WEIGHT + loss_disc * d_weight

        loss_details['loss_disc'] = loss_disc
        loss_details['loss_obj'] = loss
        loss_details['d_weight'] = d_weight
        if RRC_WEIGHT > 0:
            loss_details['recursive_loss'] = rrc_loss_details

        return loss, (loss_details, reconstruction)

    d_weight = calculate_adaptive_weight()
    grad, (loss_details, reconstruction) = compute_loss(
        state.params,
        d_weight,
        latent,
        original
    )

    # legacy code, I didn't use multi gpu
    # grad = jax.lax.pmean(grad, "batch")
    loss_details = loss_details | {"learning_rate": lr_schedule(state.step)} # type: dict
    new_state = state.apply_gradients(grads=grad)

    # metrics = jax.lax.pmean(metrics, axis_name="batch")
    return new_state, new_train_rng, loss_details, reconstruction

@jax.profiler.annotate_function
@partial(jax.jit, donate_argnums=(0, 1))
def train_step_disc(
    state_disc: TrainState,
    train_rng: jax.Array,
    original: jax.Array,
    reconstruction: jax.Array
) -> Tuple[TrainState, jax.Array, dict]:

    @jax.profiler.annotate_function
    @partial(jax.grad, has_aux=True)
    def compute_stylegan_loss(
        disc_params: dict,
        real_images: jax.Array,
        fake_images: jax.Array,
        dropout_rng: jax.Array,
        disc_model_fn: Callable[..., jax.Array]
    ) -> Tuple[jax.Array, Tuple[dict, dict, dict]]:

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

        disc_loss = disc_loss_stylegan + COST_GRAD_PENALTY * r1_grads
        disc_loss_details = {
            "pred_p_real": jnp.exp(-loss_real).mean(),  # p = 1 -> predict real is real
            "pred_p_fake": jnp.exp(-loss_fake).mean(),  # p = 1 -> predict fake is fake
            "loss_real": loss_real.mean(),
            "loss_fake": loss_fake.mean(),
            "loss_stylegan": disc_loss_stylegan,
            "loss_gradient_penalty": COST_GRAD_PENALTY * r1_grads,
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
train_ds = DecoderImageDataset(hfds.select(range(100, len(hfds))), root=DATA_ROOT) # type: ignore
test_ds = DecoderImageDataset(hfds.select(range(128)), root=DATA_ROOT) # type: ignore
if USE_WANDB:
    wandb.log({"train_dataset_size": len(train_ds)})

dataloader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=partial(DecoderImageDataset.collate_fn, return_names=False),
    num_workers=16,
    drop_last=True,
    prefetch_factor=16,
    persistent_workers=True
)

train_dl_eval = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(DecoderImageDataset.collate_fn, return_names=True),
    num_workers=4,
    drop_last=True,
    prefetch_factor=4,
    persistent_workers=True,
)
test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=partial(DecoderImageDataset.collate_fn, return_names=True),
    num_workers=4,
    drop_last=True,
    prefetch_factor=4,
    persistent_workers=True,
)

# evaluation functions

def infer_fn(batch: dict, state: TrainState) -> jax.Array:
    return reconstruct(state.params, batch["original"])

def infer_fn_ema(batch: dict, state: TrainStateEma) -> jax.Array:
    return reconstruct(state.ema_params, batch["original"])

def infer_fn_control(batch: dict) -> jax.Array:
    return reconstruct(original_params, batch["original"])

eval_batches = []
def evaluate(use_tqdm=False, step=None) -> None:
    losses = []
    losses_ema = []
    if len(eval_batches) == 0:
        iterable = test_dl if not use_tqdm else tqdm(test_dl)
        for batch in iterable:
            eval_batches.append(batch)
            print(f"Reserved {len(eval_batches)} batches for eval.")
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

def postpro(decoded_images: np.ndarray) -> list:
    """util function to postprocess images"""
    if np.any(np.isnan(decoded_images)):
        print("CRITICAL: decoded images contain NaN!")
    decoded_images = decoded_images.clip(0.0, 1.0)

    return [
        Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        for decoded_img in decoded_images
    ]

def log_images(dl: DataLoader, num_images=8, suffix="", step=None) -> None:
    logged_images = 0

    def batch_gen():
        while True:
            for batch in dl:
                yield batch

    batch_iter = batch_gen()
    while logged_images < num_images:
        batch = next(batch_iter) # type: dict

        names = batch.pop("name")
        reconstruction = infer_fn(batch, train_state)
        orig_reconstruction = infer_fn_control(batch)
        if TRAIN_EMA:
            reconstruction_ema = infer_fn_ema(batch, train_state)
            left_right = np.concatenate([batch["original"], orig_reconstruction, reconstruction, reconstruction_ema], axis=2)
        else:
            left_right = np.concatenate([batch["original"], orig_reconstruction, reconstruction], axis=2)

        images = postpro(left_right)
        if USE_WANDB:
            for name, image in zip(names, images):
                wandb.log(
                    {f"{name}{suffix}": wandb.Image(image, caption=name)}, step=step
                )
        logged_images += len(images)

def log_test_images(num_images=8, step=None) -> None:
    log_images(dl=test_dl, num_images=num_images, step=step)

def log_train_images(num_images=8, step=None) -> None:
    log_images(
        dl=train_dl_eval, num_images=num_images, suffix="|train", step=step
    )

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
def encode_latent_for_cache(original: jax.Array, sample_rng: jax.Array):
    sample_rng, new_rng = jax.random.split(sample_rng, 2)
    posterior = vae.apply( # type: ignore
        {"params": train_state.params},
        jnp.transpose(original * 2.0 - 1.0, (0, 3, 1, 2)),
        deterministic=False,
        method=vae.encode
    ) # type: FlaxAutoencoderKLOutput
    batch = {
        "original": jnp.clip(jnp.round(original * 255), 0, 255).astype(jnp.uint8),
        "latent": posterior.latent_dist.sample(sample_rng)
    }
    return batch, new_rng

@jax.jit
def vae_decode_only(latent: jax.Array):
    decoder_out = vae.apply( # type: ignore
        {"params": vae_params},
        latent,
        deterministic=False,
        method=vae.decode) # type: FlaxDecoderOutput
    return jnp.transpose(decoder_out.sample, (0, 2, 3, 1)) / 2 + 0.5

class LatentCacheDataset:
    def __init__(self, data_dir, cache_size, mmap_preload: bool = True):
        self.mmap_preload = mmap_preload
        self.file_list = []
        self.queue = Queue(64)
        for i in tqdm(range(cache_size), desc="Getting file list...", dynamic_ncols=True):
            self.file_list.append(os.path.join(data_dir, f"batch_{i}.npz"))
        self.buffer_thread = Thread(target=self.buffer_thread_function)
        self.buffer_thread.daemon = True  # Daemonize the thread to exit with the main program
        self.buffer_thread.start()
        if mmap_preload:
            self.file_mmap = []
            for filename in tqdm(self.file_list, desc="Building memory map...", dynamic_ncols=True):
                data = jnp.load(filename, mmap_mode='r')
                self.file_mmap.append(data)

    def buffer_thread_function(self):
        while True:
            for filename in self.file_list:
                file = jnp.load(filename)
                dict = {
                    'original': jnp.clip(file['original'].astype(jnp.float32) / 255, 0, 1),
                    'latent': file['latent']
                }
                self.queue.put(dict)

    def __len__(self):
        return len(self.file_list)

    # def __getitem__(self, index):
    #     if self.mmap_preload:
    #         data = self.file_mmap[index]
    #     else:
    #         data = jnp.load(self.file_list[index])
    #     return {'original': data['original'], 'latent': data['latent']}

    def __iter__(self):
        while True:
            yield self.queue.get()
        # for i in range(len(self)):
        #     yield self[i]

dataset = LatentCacheDataset('/mnt/foxhole/vae_latent_cache', 150000, mmap_preload=False)
metrics_dict = {}
metrics_list = []
steps_since_log = 0
for steps, batch in tqdm(enumerate(dataset), total=len(dataset), desc="Training...", dynamic_ncols=True):
    if PROFILE and steps == 100:
        jax.profiler.start_trace("./tensorboard")

    if LATENT_BATCH_SPLIT > 1:
        minibatches_orig = jnp.split(batch['original'], LATENT_BATCH_SPLIT, 0)
        minibatches_latent = jnp.split(batch['latent'], LATENT_BATCH_SPLIT, 0)
        metrics_batch = {} # type: dict
        for i in range(LATENT_BATCH_SPLIT):
            train_state, training_rng, metrics, fake = train_step_lc(
                train_state,
                training_rng,
                minibatches_orig[i],
                minibatches_latent[i],
                train_state_disc
            )
            # fake = vae_decode_only(batch['latent'])
            # metrics = {}
            if disc_loss_skip_schedule(train_state.step) > 0:
                train_state_disc, training_rng, metrics["disc_step"] = train_step_disc(
                    train_state_disc,
                    training_rng,
                    minibatches_orig[i],
                    fake
                )
            else:
                metrics["disc_step"] = {}

            try:
                metrics_batch = jax.tree_map(lambda x, y: x + y, metrics, metrics_batch)
            except ValueError:
                zero_tree = jax.tree_map(lambda x: 0.0, metrics) # type: dict
                zero_tree.update(metrics_batch)
                metrics_batch = jax.tree_map(lambda x, y: x + y, metrics, zero_tree)

        metrics = jax.tree_map(lambda x: x / LATENT_BATCH_SPLIT, metrics_batch)
    else:
        train_state, training_rng, metrics, fake = train_step_lc(
            train_state,
            training_rng,
            batch["original"],
            batch["latent"],
            train_state_disc
        )
        # fake = vae_decode_only(batch['latent'])
        # metrics = {}
        if disc_loss_skip_schedule(train_state.step) > 0:
            train_state_disc, training_rng, metrics["disc_step"] = train_step_disc(
                train_state_disc,
                training_rng,
                batch["original"],
                fake
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
        log_train_images(step=steps)
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


    # steps_since_log += 1
    # metrics_list.append(metrics)

    # if steps % LOG_STEPS == 1:
    #     for md in metrics_list:
    #         for key in md.keys():
    #             if isinstance(md[key], dict):
    #                 for nested_key in md[key].keys():
    #                     if not key in metrics_dict:
    #                         metrics_dict[key] = md[key]
    #                     else:
    #                         if not nested_key in metrics_dict:
    #                             metrics_dict[key][nested_key] = md[key][nested_key]
    #                         else:
    #                             metrics_dict[key][nested_key] += md[key][nested_key]
    #             else:
    #                 if not key in metrics_dict:
    #                     metrics_dict[key] = md[key]
    #                 else:
    #                     metrics_dict[key] += md[key]
    #     metrics_list = []
    #     for key in metrics_dict.keys():
    #         if isinstance(metrics_dict[key], dict):
    #             for nested_key in metrics_dict[key].keys():
    #                 metrics_dict[key][nested_key] /= steps_since_log
    #         else:
    #             metrics_dict[key] /= steps_since_log
    #     if USE_WANDB:
    #         wandb.log(metrics_dict, step=steps)
    #     metrics_dict = {}
    #     steps_since_log = 0

# metrics_dict = {}
# metrics_list = []
# steps_since_log = 0
# for steps, train_batch in zip(tqdm(range(TOTAL_STEPS)), data_iter()):
#     real = train_batch['original']

#     save_batch, training_rng = encode_latent_for_cache(
#         real,
#         training_rng
#     )
#     jnp.savez(
#         f"/mnt/foxhole/vae_latent_cache/batch_{steps}.npz",
#         original=save_batch["original"],
#         latent=save_batch["latent"]
#     )
#     # if steps == 100:
#     #     jax.profiler.start_trace("./tensorboard")
#     train_state, training_rng, metrics, fake = train_step_adaptive(
#         train_state,
#         training_rng,
#         real,
#         train_state_disc
#     )
#     if steps > DISC_LOSS_SKIP_STEPS:
#         train_state_disc, training_rng, metrics_disc = train_step_disc(
#             train_state_disc,
#             training_rng,
#             real,
#             fake
#         )
#     else:
#         metrics_disc = {}

#     # if steps == 110:
#     #     print(metrics_disc)
#     #     jax.profiler.stop_trace()
#     metrics["disc_step"] = metrics_disc

#     steps_since_log += 1
#     metrics_list.append(metrics)
#     # for key in metrics.keys():
#     #     if type(metrics[key]) == dict:
#     #         for nested_key in metrics[key].keys():
#     #             if not key in metrics_dict:
#     #                 metrics_dict[key] = metrics[key]
#     #             else:
#     #                 if not nested_key in metrics_dict:
#     #                     metrics_dict[key][nested_key] = metrics[key][nested_key]
#     #                 else:
#     #                     metrics_dict[key][nested_key] += metrics[key][nested_key]
#     #     else:
#     #         if not key in metrics_dict:
#     #             metrics_dict[key] = metrics[key]
#     #         else:
#     #             metrics_dict[key] += metrics[key]

#     if steps % LOG_STEPS == 1:
#         for md in metrics_list:
#             for key in md.keys():
#                 if isinstance(md[key], dict):
#                     for nested_key in md[key].keys():
#                         if not key in metrics_dict:
#                             metrics_dict[key] = md[key]
#                         else:
#                             if not nested_key in metrics_dict:
#                                 metrics_dict[key][nested_key] = md[key][nested_key]
#                             else:
#                                 metrics_dict[key][nested_key] += md[key][nested_key]
#                 else:
#                     if not key in metrics_dict:
#                         metrics_dict[key] = md[key]
#                     else:
#                         metrics_dict[key] += md[key]
#         metrics_list = []
#         for key in metrics_dict.keys():
#             if isinstance(metrics_dict[key], dict):
#                 for nested_key in metrics_dict[key].keys():
#                     metrics_dict[key][nested_key] /= steps_since_log
#             else:
#                 metrics_dict[key] /= steps_since_log
#         if USE_WANDB:
#             wandb.log(metrics_dict, step=steps)
#         metrics_dict = {}
#         steps_since_log = 0
#     if steps % EVAL_STEPS == 1:
#         evaluate(step=steps)
#         log_test_images(step=steps)
#         log_train_images(step=steps)
#         with Path(output_dir / "latest_state_disc.msgpack").open("wb") as f:
#             f.write(to_bytes(jax.device_get(train_state_disc)))
#         with Path(output_dir / "latest_state.msgpack").open("wb") as f:
#             f.write(to_bytes(jax.device_get(train_state)))
#         gc.collect()
#     if steps % CHECKPOINT_STEPS == 1:
#         save_checkpoint(train_state.params, steps)

# save_checkpoint(train_state.params)
