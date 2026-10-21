"""U13 reuses V24 software; its PPO initialization is independently fresh."""

from .forward_v24 import ForwardV24
from .base import install_methods, source_function


class UnifiedU13(ForwardV24):
    actor_size, critic_size = 64, 89

    def __init__(self, spec, overrides=None):
        super().__init__(spec, overrides, _profile="u13")
        source_function("SchaftUnifiedJoystickV1.__init__")(self, self._config, None)


install_methods(UnifiedU13, "u13")
