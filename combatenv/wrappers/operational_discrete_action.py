"""
Discrete action wrapper for operational-level (unit) control.

Converts discrete action indices to unit commands for the environment.

Action Space Design (19 actions per unit):
    - 0: Hold position (no waypoint)
    - 1-8: Set waypoint to 8 grid positions (3x3 around current cell)
    - 9-16: Dispatch to 8 strategic positions
    - 17: Set stance to Aggressive
    - 18: Set stance to Defensive

Action Mapping:
    0: Hold              -> Clear waypoint
    1-8: Move waypoint   -> Set waypoint to grid position
    9-16: Dispatch       -> Dispatch unit to strategic position
    17: Aggressive       -> Set stance to aggressive
    18: Defensive        -> Set stance to defensive

Grid positions (1-8, relative to unit centroid):
    [1][2][3]
    [4][X][5]  (X = current position)
    [6][7][8]

Strategic positions (9-16, 2x4 strategic cells):
    [9 ][10][11][12]
    [13][14][15][16]

Usage:
    from combatenv.wrappers import OperationalWrapper, OperationalDiscreteObsWrapper
    from combatenv.wrappers import OperationalDiscreteActionWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = OperationalWrapper(env)
    env = OperationalDiscreteObsWrapper(env, team="blue")
    env = OperationalDiscreteActionWrapper(env, team="blue")

    obs, info = env.reset()
    actions = {unit_id: 1 for unit_id in obs}  # All units move to position 1
    next_obs, rewards, terminated, truncated, info = env.step(actions)
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import gymnasium as gym

from combatenv.config import GRID_SIZE, STRATEGIC_GRID_SIZE


class OperationalDiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts discrete action indices to unit commands.

    This wrapper maps discrete actions to unit-level commands like
    setting waypoints, dispatching, and changing stances.

    Attributes:
        team: Which team to control ("blue" or "red")
        n_actions: Number of discrete actions (19)
        waypoint_step: Grid distance for waypoint movement
    """

    # Movement offsets for 8-direction waypoint grid (relative to current)
    # Positions 1-8 arranged as:
    # [1][2][3]
    # [4][X][5]
    # [6][7][8]
    WAYPOINT_OFFSETS = {
        1: (-1, -1),  # NW
        2: (0, -1),   # N
        3: (1, -1),   # NE
        4: (-1, 0),   # W
        5: (1, 0),    # E
        6: (-1, 1),   # SW
        7: (0, 1),    # S
        8: (1, 1),    # SE
    }

    # Strategic cell centers (16 cells in 4x4 grid)
    # Mapped to actions 9-16 (first 8 cells)
    # [9 ][10][11][12]
    # [13][14][15][16]

    ACTION_NAMES = [
        "Hold",
        "Waypoint NW", "Waypoint N", "Waypoint NE",
        "Waypoint W", "Waypoint E",
        "Waypoint SW", "Waypoint S", "Waypoint SE",
        "Dispatch Cell 0", "Dispatch Cell 1", "Dispatch Cell 2", "Dispatch Cell 3",
        "Dispatch Cell 4", "Dispatch Cell 5", "Dispatch Cell 6", "Dispatch Cell 7",
        "Aggressive", "Defensive",
    ]

    def __init__(self, env, team: str = "blue", waypoint_step: float = 8.0):
        """
        Initialize the operational discrete action wrapper.

        Args:
            env: Environment (should have OperationalWrapper)
            team: Which team to control ("blue" or "red")
            waypoint_step: Grid cells to move for waypoint actions
        """
        super().__init__(env)

        self.team = team
        self.waypoint_step = waypoint_step
        self.n_actions = 19

        # Expose n_states from observation wrapper if available
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states

        print(f"OperationalDiscreteActionWrapper: {self.n_actions} actions")

    def action(self, action_dict: Dict[int, int]) -> Dict[int, int]:
        """
        Convert discrete unit actions to commands.

        This method applies the commands directly to units and returns
        the original action dict for compatibility.

        Args:
            action_dict: Dict mapping unit_id -> discrete action index (0-18)

        Returns:
            Original action dict (commands are applied directly)
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{self.team}_units', [])
        unit_map = {unit.id: unit for unit in units}

        for unit_id, discrete_action in action_dict.items():
            if unit_id not in unit_map:
                continue

            unit = unit_map[unit_id]
            self._apply_unit_action(unit, discrete_action)

        return action_dict

    def _apply_unit_action(self, unit, action: int) -> None:
        """
        Apply a discrete action to a unit.

        Args:
            unit: Unit instance
            action: Discrete action index (0-18)
        """
        if action == 0:
            # Hold position - clear waypoint
            unit.clear_waypoint()

        elif 1 <= action <= 8:
            # Set relative waypoint
            offset = self.WAYPOINT_OFFSETS[action]
            centroid = unit.centroid

            new_x = centroid[0] + offset[0] * self.waypoint_step
            new_y = centroid[1] + offset[1] * self.waypoint_step

            # Clamp to grid bounds
            new_x = max(1.0, min(GRID_SIZE - 1.0, new_x))
            new_y = max(1.0, min(GRID_SIZE - 1.0, new_y))

            unit.set_waypoint(new_x, new_y)

        elif 9 <= action <= 16:
            # Dispatch to strategic cell
            cell_index = action - 9

            # Map to 4x4 grid positions (first 8 cells)
            # Actions 9-12 = top row (cells 0-3)
            # Actions 13-16 = second row (cells 4-7)
            if action <= 12:
                cell_x = action - 9
                cell_y = 0
            else:
                cell_x = action - 13
                cell_y = 1

            # Get cell center position
            cell_size = GRID_SIZE / STRATEGIC_GRID_SIZE
            target_x = (cell_x + 0.5) * cell_size
            target_y = (cell_y + 0.5) * cell_size

            # Dispatch unit (teleport if in reserve, otherwise set waypoint)
            if hasattr(unit, 'dispatch'):
                unit.dispatch((target_x, target_y))
            else:
                unit.set_waypoint(target_x, target_y)

        elif action == 17:
            # Set stance to aggressive
            unit.stance = "aggressive"

        elif action == 18:
            # Set stance to defensive
            unit.stance = "defensive"

    def step(self, action):
        """
        Step with unit actions.

        Applies unit commands then calls base environment step.
        """
        # Apply unit commands
        self.action(action)

        # Step the base environment with a no-op action
        # The controlled agent does nothing while units move via boids
        no_op = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.env.step(no_op)

    def get_action_name(self, action: int) -> str:
        """
        Get human-readable name for an action.

        Args:
            action: Discrete action index (0-18)

        Returns:
            Action name string
        """
        if 0 <= action < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action]
        return f"Unknown({action})"

    def get_waypoint_actions(self) -> List[int]:
        """
        Get list of waypoint-setting actions.

        Returns:
            List of action indices [1, 2, 3, 4, 5, 6, 7, 8]
        """
        return list(range(1, 9))

    def get_dispatch_actions(self) -> List[int]:
        """
        Get list of dispatch actions.

        Returns:
            List of action indices [9, 10, 11, 12, 13, 14, 15, 16]
        """
        return list(range(9, 17))

    def get_stance_actions(self) -> List[int]:
        """
        Get list of stance-changing actions.

        Returns:
            List of action indices [17, 18]
        """
        return [17, 18]

    def is_movement_action(self, action: int) -> bool:
        """
        Check if action involves movement (waypoint or dispatch).

        Args:
            action: Discrete action index

        Returns:
            True if action sets waypoint or dispatches
        """
        return 1 <= action <= 16

    def decode_action(self, action: int) -> Dict[str, Any]:
        """
        Decode an action into its components.

        Args:
            action: Discrete action index

        Returns:
            Dict with action type and details
        """
        if action == 0:
            return {"type": "hold"}
        elif 1 <= action <= 8:
            offset = self.WAYPOINT_OFFSETS[action]
            return {
                "type": "waypoint",
                "direction": self.ACTION_NAMES[action].replace("Waypoint ", ""),
                "offset": offset,
            }
        elif 9 <= action <= 16:
            return {
                "type": "dispatch",
                "cell": action - 9,
            }
        elif action == 17:
            return {"type": "stance", "stance": "aggressive"}
        elif action == 18:
            return {"type": "stance", "stance": "defensive"}
        else:
            return {"type": "unknown"}
