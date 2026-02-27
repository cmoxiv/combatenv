"""
Discrete action wrapper for tabular Q-learning.

Converts discrete action indices to continuous action arrays for the environment.
Actions are UNIT-RELATIVE, meaning movement is computed relative to the agent's
unit centroid, not absolute grid directions.

Action Space Design (8 actions):
    - 0: Hold (maintain relative position to unit)
    - 1: Contract (move toward unit centroid)
    - 2: Expand (move away from unit centroid)
    - 3: FlankLeft (orbit counter-clockwise around centroid)
    - 4: FlankRight (orbit clockwise around centroid)
    - 5: Shoot (stationary, engage target)
    - 6: Think (scan for enemies)
    - 7: No-op

Note: Agents do NOT shoot while moving. Shooting is a dedicated stationary action.
Movement is dynamically computed based on agent position relative to unit center:
    - Contract: Move toward centroid
    - Expand: Move away from centroid (if within UNIT_MAX_RADIUS)
    - FlankLeft: Move perpendicular (CCW) to centroid direction
    - FlankRight: Move perpendicular (CW) to centroid direction

Usage:
    from combatenv.wrappers import MultiAgentWrapper, DiscreteObservationWrapper, DiscreteActionWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)

    observations, info = env.reset()
    actions = {agent_idx: 1 for agent_idx in observations}  # All contract
    next_obs, rewards, terminated, truncated, info = env.step(actions)
"""

import math
from typing import Any, Dict, Tuple, List

import numpy as np
import gymnasium as gym

from combatenv.config import UNIT_MAX_RADIUS


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts discrete action indices to unit-relative continuous actions.

    This wrapper maps 8 discrete actions to the environment's continuous
    action space [move_x, move_y, shoot, think]. Movement directions are
    computed dynamically based on agent position relative to unit centroid.

    Note: Shooting is a dedicated stationary action. Agents do NOT shoot
    while moving - they must stop to engage targets.

    Attributes:
        n_actions: Number of discrete actions (8)
        ACTION_NAMES: List of human-readable action names
    """

    # Human-readable action names
    ACTION_NAMES = [
        "Hold",
        "Contract",
        "Expand",
        "FlankLeft",
        "FlankRight",
        "Shoot",
        "Think",
        "No-op",
    ]

    def __init__(self, env):
        """
        Initialize the discrete action wrapper.

        Args:
            env: Environment (should be DiscreteObservationWrapper)
        """
        super().__init__(env)
        self.n_actions = 8

        # Expose n_states from the observation wrapper
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states

        print(f"DiscreteActionWrapper: {self.n_actions} discrete actions (unit-relative)")

    def action(self, action_dict: Dict[int, int]) -> Dict[int, np.ndarray]:
        """
        Convert discrete actions to continuous action arrays.

        Movement is computed relative to the agent's unit centroid.

        Args:
            action_dict: Dict mapping agent_idx -> discrete action index (0-11)

        Returns:
            Dict mapping agent_idx -> continuous action array [move_x, move_y, shoot, think]
        """
        continuous_actions = {}
        base_env = self.env.unwrapped

        for agent_idx, discrete_action in action_dict.items():
            if discrete_action < 0 or discrete_action >= self.n_actions:
                raise ValueError(f"Invalid action {discrete_action}. Must be 0-{self.n_actions-1}")

            # Get the agent
            agent = self._get_agent_by_idx(base_env, agent_idx)
            if agent is None:
                # Agent not found, return no-op
                continuous_actions[agent_idx] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                continue

            # Get the agent's unit
            unit = self._get_unit_for_agent(base_env, agent)

            # Compute the continuous action
            continuous_actions[agent_idx] = self._compute_action(agent, unit, discrete_action)

        return continuous_actions

    def _get_agent_by_idx(self, base_env, agent_idx: int):
        """Get agent object by index.

        The MultiAgentWrapper uses indices into a combined list (blue + red).
        """
        # Build the combined agent list (blue first, then red)
        agent_list = list(base_env.blue_agents) + list(base_env.red_agents)

        if 0 <= agent_idx < len(agent_list):
            return agent_list[agent_idx]
        return None

    def _get_unit_for_agent(self, base_env, agent):
        """Get the unit that contains this agent."""
        # Check blue units
        if hasattr(base_env, 'blue_units'):
            for unit in base_env.blue_units:
                if agent in unit.agents:
                    return unit
        # Check red units
        if hasattr(base_env, 'red_units'):
            for unit in base_env.red_units:
                if agent in unit.agents:
                    return unit
        return None

    def _compute_action(self, agent, unit, discrete_action: int) -> np.ndarray:
        """
        Compute continuous action based on discrete action and agent's unit position.

        Args:
            agent: Agent instance
            unit: Unit instance (may be None)
            discrete_action: Discrete action index (0-7)

        Returns:
            Continuous action array [move_x, move_y, shoot, think]
        """
        # Handle Shoot action (stationary)
        if discrete_action == 5:
            return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # Handle Think action
        if discrete_action == 6:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Handle No-op action
        if discrete_action == 7:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Movement actions (0-4): no shooting while moving
        # If no unit, fall back to no movement
        if unit is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Get movement direction based on unit-relative action
        move_x, move_y = self._compute_unit_relative_movement(agent, unit, discrete_action)

        return np.array([move_x, move_y, 0.0, 0.0], dtype=np.float32)

    def _compute_unit_relative_movement(self, agent, unit, base_action: int) -> Tuple[float, float]:
        """
        Compute unit-relative movement direction.

        Args:
            agent: Agent instance
            unit: Unit instance
            base_action: Base action index (0-4)

        Returns:
            (move_x, move_y) normalized direction
        """
        # Get vector from agent to unit centroid
        centroid = unit.centroid
        agent_x, agent_y = agent.position
        dx = agent_x - centroid[0]
        dy = agent_y - centroid[1]
        dist = math.sqrt(dx * dx + dy * dy)

        # Avoid division by zero
        if dist < 0.1:
            # Agent is at centroid, default to no movement for most actions
            if base_action == 2:  # Expand
                # Default to moving in a direction based on agent position hash
                # Use agent's position as a pseudo-ID for deterministic spread
                pseudo_id = hash((agent_x, agent_y, agent.orientation))
                angle = (pseudo_id * 137.5) % 360  # Golden angle for good spread
                return (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            return (0.0, 0.0)

        # Normalize the direction vector
        nx = dx / dist
        ny = dy / dist

        if base_action == 0:  # Hold
            return (0.0, 0.0)

        elif base_action == 1:  # Contract (toward centroid)
            return (-nx, -ny)

        elif base_action == 2:  # Expand (away from centroid)
            # Only expand if within allowed radius
            if dist < UNIT_MAX_RADIUS - 1.0:
                return (nx, ny)
            return (0.0, 0.0)  # At max radius, don't expand further

        elif base_action == 3:  # FlankLeft (CCW around centroid)
            # Perpendicular: rotate 90 degrees counter-clockwise
            return (-ny, nx)

        elif base_action == 4:  # FlankRight (CW around centroid)
            # Perpendicular: rotate 90 degrees clockwise
            return (ny, -nx)

        return (0.0, 0.0)

    def reverse_action(self, continuous_action: np.ndarray) -> int:
        """
        Convert continuous action back to discrete index (for debugging).

        Note: This is approximate for unit-relative actions since the
        exact mapping depends on agent position.

        Args:
            continuous_action: [move_x, move_y, shoot, think] array

        Returns:
            Approximate discrete action index (0-7)
        """
        move_x, move_y, shoot, think = continuous_action[:4]

        # Check for Think action
        if think > 0.5:
            return 6  # Think

        # Check for Shoot action (stationary)
        if abs(move_x) < 0.1 and abs(move_y) < 0.1 and shoot > 0.5:
            return 5  # Shoot

        # Check for No-op action
        if abs(move_x) < 0.1 and abs(move_y) < 0.1:
            return 7  # No-op

        # For actual movement, we can't determine the exact action without
        # knowing the agent's position relative to unit. Return a guess.
        return 1  # Default to Contract

    def get_action_name(self, action: int) -> str:
        """
        Get human-readable name for an action.

        Args:
            action: Discrete action index (0-7)

        Returns:
            Action name string
        """
        if 0 <= action < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action]
        return f"Unknown({action})"

    def get_movement_actions(self) -> List[int]:
        """
        Get list of movement actions.

        Returns:
            List of action indices [1, 2, 3, 4]
        """
        return [1, 2, 3, 4]

    def get_shoot_action(self) -> int:
        """
        Get the shoot action index.

        Returns:
            Action index 5
        """
        return 5

    def get_hold_action(self) -> int:
        """
        Get the hold action index.

        Returns:
            Action index 0
        """
        return 0

    def does_action_shoot(self, action: int) -> bool:
        """
        Check if an action includes shooting.

        Args:
            action: Discrete action index (0-7)

        Returns:
            True if action is Shoot (5)
        """
        return action == 5

    def is_think_action(self, action: int) -> bool:
        """
        Check if action is the Think action.

        Args:
            action: Discrete action index (0-7)

        Returns:
            True if action is Think (6)
        """
        return action == 6

    def is_no_action(self, action: int) -> bool:
        """
        Check if action is No-op action.

        Args:
            action: Discrete action index (0-7)

        Returns:
            True if action is No-op (7)
        """
        return action == 7

    def is_movement_action(self, action: int) -> bool:
        """
        Check if action involves movement.

        Args:
            action: Discrete action index (0-7)

        Returns:
            True if action involves movement (1-4)
        """
        return 1 <= action <= 4

    def decode_action(self, action: int) -> Dict[str, Any]:
        """
        Decode an action into its components.

        Args:
            action: Discrete action index

        Returns:
            Dict with action type and details
        """
        if action == 5:
            return {"type": "shoot", "movement": None, "shoot": True}
        if action == 6:
            return {"type": "think", "movement": None, "shoot": False}
        if action == 7:
            return {"type": "no_op", "movement": None, "shoot": False}

        movement_types = ["hold", "contract", "expand", "flank_left", "flank_right"]

        return {
            "type": "tactical",
            "movement": movement_types[action],
            "shoot": False,
        }
