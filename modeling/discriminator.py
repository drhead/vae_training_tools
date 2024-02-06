import jax
import jax.numpy as jnp
from typing import Sequence, Union, Tuple
import flax.linen as nn
from transformers import PretrainedConfig
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from flax.core.frozen_dict import FrozenDict

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
        # self.norm = nn.BatchNorm(momentum=0.9)

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
        params: dict | None = None,
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
