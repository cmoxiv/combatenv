"""
Discrete action wrapper for strategic-level (grid) control.

Converts discrete action indices to strategic commands for unit allocation.

Action Space Design (25 actions):
    - 0: Hold current allocation
    - 1-16: Focus forces on strategic cell (0-15)
    - 17-20: Aggressive push in direction (N/S/E/W)
    - 21-24: Defensive stance in quadrant (TL/TR/BL/BR)

Action Mapping:
    0: Hold              -> No change to unit objectives
    1-16: Focus Cell     -> Set waypoints for nearest units to target cell
    17: Push North       -> Units advance northward
    18: Push South       -> Units advance southward
    19: Push East        -> Units advance eastward
    20: Push West        -> Units advance westward
    21: Defend Top-Left  -> Units defend quadrant
    22: Defend Top-Right -> Units defend quadrant
    23: Defend Bot-Left  -> Units defend quadrant
    24: Defend Bot-Right -> Units defend quadrant

Usage:
    from combatenv.wrappers import StrategicWrapper, StrategicDiscreteObsWrapper
    from combatenv.wrappers import StrategicDiscreteActionWrapper

    env = TacticalCombatEnv(render_mode=None)
    env = StrategicWrapper(env)
    env = StrategicDiscreteObsWrapper(env)
    env = StrategicDiscreteActionWrapper(env, team="blue")

    obs, info = env.reset()
    action = 5  # Focus on cell 4
    next_obs, reward, terminated, truncated, info = env.step(action)
"""

from typing import Any, Dict, List

import numpy as np
import gymnasium as gym

from combatenv.config import GRID_SIZE, STRATEGIC_GRID_SIZE


class StrategicDiscreteActionWrapper(gym.ActionWrapper):
    """
    Converts discrete action indices to strategic unit commands.

    This wrapper maps strategic-level decisions to unit-level objectives,
    coordinating multiple units toward strategic goals.

    Attributes:
        team: Which team to control ("blue" or "red")
        n_actions: Number of discrete actions (25)
        units_per_command: Number of units to assign per command
    """

    ACTION_NAMES = [
        "Hold",
        "Focus Cell 0", "Focus Cell 1", "Focus Cell 2", "Focus Cell 3",
        "Focus Cell 4", "Focus Cell 5", "Focus Cell 6", "Focus Cell 7",
        "Focus Cell 8", "Focus Cell 9", "Focus Cell 10", "Focus Cell 11",
        "Focus Cell 12", "Focus Cell 13", "Focus Cell 14", "Focus Cell 15",
        "Push North", "Push South", "Push East", "Push West",
        "Defend TL", "Defend TR", "Defend BL", "Defend BR",
    ]

    # Push directions: (dx, dy) offsets in grid cells
    PUSH_DIRECTIONS = {
        17: (0, -1),   # North
        18: (0, 1),    # South
        19: (1, 0),    # East
        20: (-1, 0),   # West
    }

    # Quadrant centers for defensive stances
    QUADRANT_CENTERS = {
        21: (0.25, 0.25),  # Top-left
        22: (0.75, 0.25),  # Top-right
        23: (0.25, 0.75),  # Bottom-left
        24: (0.75, 0.75),  # Bottom-right
    }

    def __init__(self, env, team: str = "blue", units_per_command: int = 2):
        """
        Initialize the strategic discrete action wrapper.

        Args:
            env: Environment (should have StrategicWrapper)
            team: Which team to control ("blue" or "red")
            units_per_command: Number of units to assign per strategic action
        """
        super().__init__(env)

        self.team = team
        self.units_per_command = units_per_command
        self.n_actions = 25

        # Expose n_states from observation wrapper if available
        if hasattr(env, 'n_states'):
            self.n_states = env.n_states

        # Cell size for coordinate conversion
        self.cell_size = GRID_SIZE / STRATEGIC_GRID_SIZE

        print(f"StrategicDiscreteActionWrapper: {self.n_actions} actions")

    def action(self, action: int) -> int:
        """
        Convert discrete strategic action to unit commands.

        This method applies strategic commands directly to units.

        Args:
            action: Discrete action index (0-24)

        Returns:
            Original action (commands applied directly)
        """
        if action == 0:
            # Hold - no change
            pass

        elif 1 <= action <= 16:
            # Focus on strategic cell
            cell_index = action - 1
            self._focus_on_cell(cell_index)

        elif 17 <= action <= 20:
            # Push in direction
            self._push_direction(action)

        elif 21 <= action <= 24:
            # Defend quadrant
            self._defend_quadrant(action)

        return action

    def _focus_on_cell(self, cell_index: int) -> None:
        """
        Direct units to focus on a strategic cell.

        Args:
            cell_index: Strategic cell index (0-15)
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{self.team}_units', [])

        if not units:
            return

        # Get cell center position
        cell_x = cell_index % STRATEGIC_GRID_SIZE
        cell_y = cell_index // STRATEGIC_GRID_SIZE
        target_x = (cell_x + 0.5) * self.cell_size
        target_y = (cell_y + 0.5) * self.cell_size

        # Find nearest available units
        available_units = [u for u in units if u.alive_count > 0]

        # Sort by distance to target
        available_units.sort(key=lambda u: (
            (u.centroid[0] - target_x) ** 2 +
            (u.centroid[1] - target_y) ** 2
        ))

        # Assign nearest units
        for unit in available_units[:self.units_per_command]:
            # Add some spread to avoid clumping
            offset_x = (hash(unit.id) % 5 - 2) * 2.0
            offset_y = (hash(unit.id * 7) % 5 - 2) * 2.0

            unit.set_waypoint(
                target_x + offset_x,
                target_y + offset_y
            )
            unit.stance = "aggressive"

    def _push_direction(self, action: int) -> None:
        """
        Push units in a direction.

        Args:
            action: Push action (17-20)
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{self.team}_units', [])

        if not units:
            return

        dx, dy = self.PUSH_DIRECTIONS[action]
        push_distance = self.cell_size  # One strategic cell

        for unit in units:
            if unit.alive_count == 0:
                continue

            centroid = unit.centroid
            new_x = centroid[0] + dx * push_distance
            new_y = centroid[1] + dy * push_distance

            # Clamp to grid
            new_x = max(2.0, min(GRID_SIZE - 2.0, new_x))
            new_y = max(2.0, min(GRID_SIZE - 2.0, new_y))

            unit.set_waypoint(new_x, new_y)
            unit.stance = "aggressive"

    def _defend_quadrant(self, action: int) -> None:
        """
        Set units to defend a quadrant.

        Args:
            action: Defend action (21-24)
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{self.team}_units', [])

        if not units:
            return

        # Get quadrant center (normalized 0-1)
        norm_x, norm_y = self.QUADRANT_CENTERS[action]
        target_x = norm_x * GRID_SIZE
        target_y = norm_y * GRID_SIZE

        # Assign units to defend
        for i, unit in enumerate(units):
            if unit.alive_count == 0:
                continue

            # Spread units around quadrant center
            angle = (i / max(1, len(units))) * 2 * np.pi
            radius = self.cell_size * 0.5
            offset_x = np.cos(angle) * radius
            offset_y = np.sin(angle) * radius

            unit.set_waypoint(
                target_x + offset_x,
                target_y + offset_y
            )
            unit.stance = "defensive"

    def step(self, action):
        """
        Step with strategic action.

        Applies strategic commands then calls base environment step.
        """
        # Apply strategic command
        self.action(action)

        # Step the base environment with a no-op action
        no_op = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.env.step(no_op)

    def get_action_name(self, action: int) -> str:
        """
        Get human-readable name for an action.

        Args:
            action: Discrete action index (0-24)

        Returns:
            Action name string
        """
        if 0 <= action < len(self.ACTION_NAMES):
            return self.ACTION_NAMES[action]
        return f"Unknown({action})"

    def get_focus_actions(self) -> List[int]:
        """
        Get list of cell focus actions.

        Returns:
            List of action indices [1, ..., 16]
        """
        return list(range(1, 17))

    def get_push_actions(self) -> List[int]:
        """
        Get list of push actions.

        Returns:
            List of action indices [17, 18, 19, 20]
        """
        return list(range(17, 21))

    def get_defend_actions(self) -> List[int]:
        """
        Get list of defend actions.

        Returns:
            List of action indices [21, 22, 23, 24]
        """
        return list(range(21, 25))

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
        elif 1 <= action <= 16:
            cell = action - 1
            return {
                "type": "focus",
                "cell": cell,
                "cell_x": cell % STRATEGIC_GRID_SIZE,
                "cell_y": cell // STRATEGIC_GRID_SIZE,
            }
        elif 17 <= action <= 20:
            directions = {17: "north", 18: "south", 19: "east", 20: "west"}
            return {
                "type": "push",
                "direction": directions[action],
            }
        elif 21 <= action <= 24:
            quadrants = {21: "top_left", 22: "top_right", 23: "bottom_left", 24: "bottom_right"}
            return {
                "type": "defend",
                "quadrant": quadrants[action],
            }
        else:
            return {"type": "unknown"}

    def get_cell_for_action(self, action: int) -> int:
        """
        Get strategic cell index for a focus action.

        Args:
            action: Discrete action index

        Returns:
            Cell index (0-15) or -1 if not a focus action
        """
        if 1 <= action <= 16:
            return action - 1
        return -1
