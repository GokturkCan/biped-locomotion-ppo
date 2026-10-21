# Adapted for the modular research release; behavior changes are documented in the extraction manifest.
# Copyright 2025 DeepMind Technologies Limited
# Licensed under Apache-2.0; see LICENSE.
# Selected MuJoCo Playground MJX auto-reset wrapper, no assets or randomization.
from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jp
from biped_locomotion.environments import mjx_support as mjx_env


class Wrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)

    @property
    def unwrapped(self):
        return self.env.unwrapped


class BraxAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done.

    If `full_reset` is disabled (default):
      * the environment will reset to a cached first state.
      * only data and obs are reset, not the environment info.

    If `full_reset` is enabled:
      * the environment will call env.reset during env.step on done.
      * `full_reset` will thus incur a penalty in wallclock time depending on the
        complexity of the reset function.
      * info is fully reset, except for info under the key
        `AutoResetWrapper_preserve_info`, which is passed through from the prior
        step. This can be used for curriculum learning.

    Attributes:
      env: The wrapped environment.
      full_reset: Whether to call `env.reset` during `env.step` on done.
    """

    def __init__(self, env: Any, full_reset: bool = False):
        super().__init__(env)
        self._full_reset = full_reset
        self._info_key = "AutoResetWrapper"

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng_key = jax.vmap(jax.random.split)(rng)
        rng, key = (rng_key[..., 0], rng_key[..., 1])
        state = self.env.reset(key)
        state.info[f"{self._info_key}_first_data"] = state.data
        state.info[f"{self._info_key}_first_obs"] = state.obs
        state.info[f"{self._info_key}_rng"] = rng
        state.info[f"{self._info_key}_done_count"] = jp.zeros(key.shape[:-1], dtype=int)
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        reset_state = None
        rng_key = jax.vmap(jax.random.split)(state.info[f"{self._info_key}_rng"])
        reset_rng, reset_key = (rng_key[..., 0], rng_key[..., 1])
        if self._full_reset:
            reset_state = self.reset(reset_key)
            reset_data = reset_state.data
            reset_obs = reset_state.obs
        else:
            reset_data = state.info[f"{self._info_key}_first_data"]
            reset_obs = state.info[f"{self._info_key}_first_obs"]
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, reset_data, state.data)
        obs = jax.tree.map(where_done, reset_obs, state.obs)
        next_info = state.info
        done_count_key = f"{self._info_key}_done_count"
        if self._full_reset and reset_state:
            next_info = jax.tree.map(where_done, reset_state.info, state.info)
            next_info[done_count_key] = state.info[done_count_key]
            if "steps" in next_info:
                next_info["steps"] = state.info["steps"]
            preserve_info_key = f"{self._info_key}_preserve_info"
            if preserve_info_key in next_info:
                next_info[preserve_info_key] = state.info[preserve_info_key]
        next_info[done_count_key] += state.done.astype(int)
        next_info[f"{self._info_key}_rng"] = reset_rng
        return state.replace(data=data, obs=obs, info=next_info)


def wrap_for_training(env, episode_length=1000, action_repeat=1, randomization_fn=None):
    if randomization_fn is not None:
        raise ValueError("Domain randomization is outside this release contract")
    from brax.envs.wrappers.training import VmapWrapper, EpisodeWrapper

    return BraxAutoResetWrapper(
        EpisodeWrapper(VmapWrapper(env), episode_length, action_repeat),
        full_reset=False,
    )
