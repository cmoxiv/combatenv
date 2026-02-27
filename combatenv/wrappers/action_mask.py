"""
Action mask wrapper to disable certain actions.

When a disabled action is chosen, it is replaced with a randomly sampled
valid action. This is useful for curriculum learning or restricting
complex behaviors.

Usage:
    from combatenv.wrappers import ActionMaskWrapper

    env = DiscreteActionWrapper(env)
    env = ActionMaskWrapper(env, disabled_actions=[4, 5, 6, 7])  # Disable move+shoot

    # If agent selects action 5 (South+Shoot), it will be replaced with
    # a random valid action from [0, 1, 2, 3]
"""

from typing import Dict, List, Optional, Set, Any

import numpy as np
import gymnasium as gym


class ActionMaskWrapper(gym.Wrapper):
    """
    Wrapper that disables certain actions by replacing them with random valid actions.

    When an agent selects a disabled action, it is replaced with a uniformly
    sampled action from the set of valid (enabled) actions.

    Attributes:
        disabled_actions: Set of action indices that are disabled
        valid_actions: List of action indices that are enabled
        n_actions: Total number of actions
        n_states: Number of discrete states (passed through from inner wrapper)
    """

    def __init__(self, env, disabled_actions: Optional[List[int]] = None):
        """
        Initialize the action mask wrapper.

        Args:
            env: Environment (should be DiscreteActionWrapper or similar)
            disabled_actions: List of action indices to disable. If None, no actions disabled.
        """
        super().__init__(env)

        self.n_actions = env.n_actions
        self.n_states = env.n_states

        # Set up disabled/valid actions
        self.disabled_actions: Set[int] = set(disabled_actions or [])
        self.valid_actions: List[int] = [
            a for a in range(self.n_actions) if a not in self.disabled_actions
        ]

        if not self.valid_actions:
            raise ValueError("Cannot disable all actions - at least one must be valid")

        # Statistics
        self.replacements_made = 0

        # Print configuration
        if self.disabled_actions:
            disabled_names = [env.get_action_name(a) for a in sorted(self.disabled_actions)]
            print(f"ActionMaskWrapper: Disabled {len(self.disabled_actions)} actions")
            print(f"  Disabled: {disabled_names}")
            print(f"  Valid: {self.valid_actions}")
        else:
            print("ActionMaskWrapper: No actions disabled")

    def step(self, action: Dict[int, int]) -> Any:
        """
        Replace disabled actions with random valid actions, then step.

        Args:
            action: Dict mapping agent_idx -> discrete action index

        Returns:
            Standard gym step tuple (obs, reward, terminated, truncated, info)
        """
        masked_actions = {}
        for agent_idx, act in action.items():
            if act in self.disabled_actions:
                # Replace with random valid action
                masked_actions[agent_idx] = int(np.random.choice(self.valid_actions))
                self.replacements_made += 1
            else:
                masked_actions[agent_idx] = act

        # Pass through to inner wrapper
        return self.env.step(masked_actions)

    def get_valid_actions(self) -> List[int]:
        """Get list of valid (enabled) action indices."""
        return self.valid_actions.copy()

    def get_disabled_actions(self) -> List[int]:
        """Get list of disabled action indices."""
        return list(self.disabled_actions)

    def is_action_valid(self, action: int) -> bool:
        """Check if an action is valid (not disabled)."""
        return action not in self.disabled_actions

    def get_statistics(self) -> Dict:
        """Get wrapper statistics."""
        return {
            "disabled_actions": list(self.disabled_actions),
            "valid_actions": self.valid_actions,
            "replacements_made": self.replacements_made,
        }

    def reset_statistics(self) -> None:
        """Reset replacement counter."""
        self.replacements_made = 0
