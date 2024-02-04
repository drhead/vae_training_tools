import jax
import flax
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training.train_state import TrainState
from copy import deepcopy
import optax
from typing import Any, List

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

class TrainStateEmaGamma(TrainState):
    # Alternative EMA tracking proposed by Karras, et al. Dec 2023
    ema_params: List[flax.core.FrozenDict[str, Any]] = flax.struct.field(pytree_node=True)
    ema_gammas: List[float] = [6.94, 16.97]

    @classmethod
    def create(cls, *, apply_fn, params, tx, ema_gammas, **kwargs):
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
            ema_params=[deepcopy(params) for _ in ema_gammas],
            ema_gammas=ema_gammas,
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

        new_ema_params = []
        for idx, gamma in enumerate(self.ema_gammas):
            decay = (1 - 1/self.step) ** (gamma + 1)
            new_ema_params[idx] = jax.tree_map(
                lambda ema, param: decay * ema + (1.0 - decay) * param,
                self.ema_params[idx],
                new_params
            )

        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            **kwargs,
        )