"""
Action suppression wrapper for replacing suppressed actions with random alternatives.

Use this wrapper to disable specific discrete actions during tactical training.
When a suppressed action is received, it gets replaced with a random non-suppressed action.

This wrapper does NOT modify agent attributes (health, is_alive, cooldowns, etc).

Usage:
    from combatenv.wrappers import ActionSuppressionWrapper, DiscreteActionWrapper

    env = TacticalCombatEnv(render_mode="human")
    env = DiscreteActionWrapper(env)
    env = ActionSuppressionWrapper(env, suppress=["shoot"])

    # Now when action 5 (shoot) is taken, it gets replaced with a random
    # action from [0, 1, 2, 3, 4, 6, 7]
"""

import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import gymnasium as gym


# Tactical action name to index mapping (DiscreteActionWrapper)
ACTION_NAMES = {
    "hold": 0,
    "contract": 1,
    "expand": 2,
    "flank_left": 3,
    "flank_right": 4,
    "shoot": 5,
    "think": 6,
    "noop": 7,
}


class ActionSuppressionWrapper(gym.ActionWrapper):
    """
    Replace suppressed actions with random alternatives.

    Does NOT modify agent attributes (health, is_alive, cooldowns, etc).
    Works at the tactical level with DiscreteActionWrapper.

    Attributes:
        suppress_actions: Set of action indices to suppress
        allowed_actions: List of action indices that are allowed
    """

    def __init__(self, env, suppress: List[str] = None):
        """
        Initialize the action suppression wrapper.

        Args:
            env: Environment to wrap (should have DiscreteActionWrapper)
            suppress: List of action names to suppress, e.g. ["shoot", "think"]
                     Valid names: hold, contract, expand, flank_left, flank_right,
                                 shoot, think, noop
        """
        super().__init__(env)

        # Convert action names to indices
        suppress = suppress or []
        self.suppress_actions = set()
        for name in suppress:
            name_lower = name.lower()
            if name_lower in ACTION_NAMES:
                self.suppress_actions.add(ACTION_NAMES[name_lower])
            else:
                raise ValueError(
                    f"Unknown action: {name}. Valid actions: {list(ACTION_NAMES.keys())}"
                )

        # Build allowed actions list
        self.all_actions = set(range(8))
        self.allowed_actions = list(self.all_actions - self.suppress_actions)

        if not self.allowed_actions:
            raise ValueError("Cannot suppress all actions - at least one must be allowed")

        print(f"ActionSuppressionWrapper: suppressing {suppress}")

    def action(self, action: Union[int, np.ndarray, Dict]) -> Union[int, np.ndarray, Dict]:
        """
        Replace suppressed actions with random alternatives.

        Args:
            action: Single action (int), array, or dict of actions (multi-agent)

        Returns:
            Modified action with suppressed actions replaced
        """
        # Handle dict of actions (multi-agent)
        if isinstance(action, dict):
            return {
                agent_id: self._replace_if_suppressed(a)
                for agent_id, a in action.items()
            }

        # Handle numpy array (convert to int)
        if isinstance(action, np.ndarray):
            action = int(action.item()) if action.ndim == 0 else int(action[0])

        # Handle single action
        return self._replace_if_suppressed(action)

    def _replace_if_suppressed(self, action: int) -> int:
        """
        Replace action if it's in the suppress list.

        Args:
            action: Action index

        Returns:
            Original action if allowed, or random allowed action if suppressed
        """
        action_int = int(action)
        if action_int in self.suppress_actions:
            return random.choice(self.allowed_actions)
        return action_int
