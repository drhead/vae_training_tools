import jax
import jax.numpy as jnp
import jaxwt as jwt

from functools import partial

@partial(jax.jit, static_argnames="axis")
def kurtosis(x: jax.Array, axis=None) -> jax.Array:
    epsilon = 1e-10
    mean_squared = jnp.mean(x ** 2, axis=axis)
    mean_squared = jnp.where(mean_squared < epsilon, mean_squared + epsilon, mean_squared)
    kurt = (jnp.mean(x ** 4, axis=axis) / mean_squared ** 2) - 3
    return kurt

@partial(jax.jit, static_argnames="axis")
def real_softmax(x: jax.Array, axis=None) -> jax.Array:
    return jax.scipy.special.logsumexp(x, axis=axis)

@partial(jax.jit, static_argnames="axis")
def real_softmin(x: jax.Array, axis=None) -> jax.Array:
    return -jax.scipy.special.logsumexp(-x, axis=axis)

def kurtosis_concentration(x: jax.Array) -> jax.Array:
    coeffs_kt = jnp.zeros((20, x.shape[0]))
    # NOTE: Due to shape mismatches this can't be turned into lax
    for i in range(1, 28):
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

@partial(jax.jit, static_argnames=['alpha', 'beta', 'gamma'])
def sigmoid_mask(x: jax.Array, alpha: float, beta: float, gamma: float) -> jax.Array:
    # alpha - midpoint of sigmod
    # beta  - position where mask = 0.01
    # gamma - maximum value

    k = -jnp.log(99) / (beta - alpha)
    return gamma / (1 + jnp.exp(-k * (x - alpha)))
