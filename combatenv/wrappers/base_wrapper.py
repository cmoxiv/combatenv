"""
BaseWrapper - Foundation class with attribute forwarding for wrapper stack.

Gymnasium's gym.Wrapper doesn't automatically forward attribute access to
inner environments. This base class adds __getattr__ to enable attribute
forwarding through the wrapper chain.

Usage:
    class MyWrapper(BaseWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.my_attribute = "value"
"""

from typing import Any
import gymnasium as gym


class BaseWrapper(gym.Wrapper):
    """
    Base wrapper class with attribute forwarding.

    Attributes defined on inner wrappers are accessible through outer wrappers
    via the __getattr__ method.

    Example:
        env = GridWorld()
        env = AgentWrapper(env)  # defines 'agents'
        env = TeamWrapper(env)   # can access env.agents
    """

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the wrapped environment.

        This allows attributes defined on inner wrappers to be accessible
        from outer wrappers.

        Args:
            name: Attribute name to look up

        Returns:
            The attribute value from the wrapped environment

        Raises:
            AttributeError: If the attribute is not found
        """
        # Don't forward private attributes
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Forward to wrapped environment
        return getattr(self.env, name)
