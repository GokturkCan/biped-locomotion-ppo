"""Canonical V24 algorithm, requiring an authorized external model."""

from .base import ExternalModelEnvironment, install_methods


class ForwardV24(ExternalModelEnvironment):
    actor_size, critic_size = 57, 82

    def __init__(self, spec, overrides=None, *, _profile="v24"):
        super().__init__(spec, _profile, overrides)


install_methods(ForwardV24, "v24")
