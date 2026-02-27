"""
Strategic wrapper for high-level 4x4 grid observation and control.

This wrapper provides strategic-level capabilities:
- 4x4 strategic grid observation (terrain + occupancy)
- Strategic cell queries (which cell contains an agent/unit)
- Waypoint index to position conversion
- Unit-level strategic observations
- Strategic grid visualization toggle

The strategic grid divides the 64x64 tactical grid into a 4x4 grid
where each strategic cell covers 16x16 tactical cells.

Usage:
    from combatenv.wrappers import StrategicWrapper

    env = TacticalCombatEnv(render_mode="human")
    env = OperationalWrapper(env)
    env = StrategicWrapper(env)

    # Get strategic observation
    obs = env.get_strategic_observation()

    # Get cell containing a unit
    cell = env.get_unit_strategic_cell(unit_id, "blue")

    # Convert waypoint index to position
    x, y = env.waypoint_index_to_position(42)
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from combatenv.config import (
    GRID_SIZE,
    STRATEGIC_GRID_SIZE,
    TACTICAL_CELLS_PER_STRATEGIC,
)
from combatenv.strategic_grid import (
    build_strategic_observation,
    get_terrain_board,
    get_occupancy_board,
    get_major_terrain,
    get_major_occupancy,
    strategic_cell_to_position,
    position_to_strategic_cell,
    get_strategic_cell_bounds,
)


# Unit observation size: 7 floats per unit
UNIT_OBS_SIZE = 7

# Waypoint grid discretization: 8x8 = 64 possible positions
WAYPOINT_GRID_SIZE = 8


class StrategicWrapper(gym.Wrapper):
    """
    Wrapper for strategic-level 4x4 grid observation and control.

    Provides high-level observation and utility methods for strategic
    agents that need to reason about the battlefield at a macro level.

    Attributes:
        show_strategic_grid: Whether strategic grid overlay is visible
        waypoint_grid_size: Size of waypoint discretization grid
    """

    def __init__(
        self,
        env,
        show_strategic_grid: bool = True,
        waypoint_grid_size: int = 8,
    ):
        """
        Initialize the strategic wrapper.

        Args:
            env: Base environment to wrap
            show_strategic_grid: Initial state of grid overlay
            waypoint_grid_size: Size of waypoint grid (default 8x8)
        """
        super().__init__(env)

        self.show_strategic_grid = show_strategic_grid
        self.waypoint_grid_size = waypoint_grid_size

        # Sync with base env
        self._sync_to_base_env()

    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset and sync strategic grid state."""
        result = self.env.reset(**kwargs)
        self._sync_to_base_env()
        return result

    def _sync_to_base_env(self) -> None:
        """Sync strategic grid visibility to base env."""
        base_env = self.env.unwrapped
        if hasattr(base_env, 'show_strategic_grid'):
            base_env.show_strategic_grid = self.show_strategic_grid

    def render(self):
        """Render with strategic grid overlay sync."""
        self._sync_to_base_env()
        return self.env.render()

    # ==================== Strategic Observation ====================

    def get_strategic_observation(self) -> np.ndarray:
        """
        Build 32-value strategic observation array.

        The observation consists of:
        - First 16 values: Major terrain type per strategic cell
        - Last 16 values: Major occupancy per strategic cell

        Returns:
            numpy array of shape (32,) with values in [0, 1]
        """
        base_env = self.env.unwrapped
        terrain_grid = getattr(base_env, 'terrain_grid', None)
        blue_agents = getattr(base_env, 'blue_agents', [])
        red_agents = getattr(base_env, 'red_agents', [])

        if terrain_grid is None:
            return np.zeros(32, dtype=np.float32)

        return build_strategic_observation(terrain_grid, blue_agents, red_agents)

    def get_terrain_board(self) -> np.ndarray:
        """
        Get 4x4 terrain board as 2D array.

        Returns:
            numpy array of shape (4, 4) with terrain values (0-1)
        """
        base_env = self.env.unwrapped
        terrain_grid = getattr(base_env, 'terrain_grid', None)

        if terrain_grid is None:
            return np.zeros((STRATEGIC_GRID_SIZE, STRATEGIC_GRID_SIZE), dtype=np.float32)

        return get_terrain_board(terrain_grid)

    def get_occupancy_board(self) -> np.ndarray:
        """
        Get 4x4 occupancy board as 2D array.

        Values: 0.0 (blue dominant), 0.5 (neutral), 1.0 (red dominant)

        Returns:
            numpy array of shape (4, 4) with occupancy values
        """
        base_env = self.env.unwrapped
        blue_agents = getattr(base_env, 'blue_agents', [])
        red_agents = getattr(base_env, 'red_agents', [])

        return get_occupancy_board(blue_agents, red_agents)

    def get_cell_terrain(self, cell_x: int, cell_y: int) -> float:
        """
        Get major terrain type for a strategic cell.

        Args:
            cell_x: Strategic cell X (0-3)
            cell_y: Strategic cell Y (0-3)

        Returns:
            Normalized terrain value (0-1)
        """
        base_env = self.env.unwrapped
        terrain_grid = getattr(base_env, 'terrain_grid', None)

        if terrain_grid is None:
            return 0.0

        return get_major_terrain(terrain_grid, cell_x, cell_y)

    def get_cell_occupancy(self, cell_x: int, cell_y: int) -> float:
        """
        Get occupancy for a strategic cell.

        Args:
            cell_x: Strategic cell X (0-3)
            cell_y: Strategic cell Y (0-3)

        Returns:
            0.0 (blue), 0.5 (neutral), 1.0 (red)
        """
        base_env = self.env.unwrapped
        blue_agents = getattr(base_env, 'blue_agents', [])
        red_agents = getattr(base_env, 'red_agents', [])

        return get_major_occupancy(blue_agents, red_agents, cell_x, cell_y)

    # ==================== Coordinate Conversion ====================

    def position_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert tactical position to strategic cell coordinates.

        Args:
            x: X position in tactical grid
            y: Y position in tactical grid

        Returns:
            (cell_x, cell_y) strategic cell coordinates
        """
        cell_index = position_to_strategic_cell(x, y)
        cell_x = cell_index % STRATEGIC_GRID_SIZE
        cell_y = cell_index // STRATEGIC_GRID_SIZE
        return (cell_x, cell_y)

    def position_to_cell_index(self, x: float, y: float) -> int:
        """
        Convert tactical position to strategic cell index.

        Args:
            x: X position in tactical grid
            y: Y position in tactical grid

        Returns:
            Cell index (0-15)
        """
        return position_to_strategic_cell(x, y)

    def cell_to_position(self, cell_x: int, cell_y: int) -> Tuple[float, float]:
        """
        Convert strategic cell to tactical grid center position.

        Args:
            cell_x: Strategic cell X (0-3)
            cell_y: Strategic cell Y (0-3)

        Returns:
            (x, y) center position in tactical grid
        """
        cell_index = cell_y * STRATEGIC_GRID_SIZE + cell_x
        return strategic_cell_to_position(cell_index)

    def cell_index_to_position(self, cell_index: int) -> Tuple[float, float]:
        """
        Convert strategic cell index to tactical grid center position.

        Args:
            cell_index: Cell index (0-15)

        Returns:
            (x, y) center position in tactical grid
        """
        return strategic_cell_to_position(cell_index)

    def get_cell_bounds(self, cell_index: int) -> Tuple[int, int, int, int]:
        """
        Get tactical grid bounds for a strategic cell.

        Args:
            cell_index: Strategic cell index (0-15)

        Returns:
            (x_start, y_start, x_end, y_end) in tactical coordinates
        """
        return get_strategic_cell_bounds(cell_index)

    # ==================== Waypoint Grid ====================

    def waypoint_index_to_position(self, waypoint_index: int) -> Tuple[float, float]:
        """
        Convert waypoint index to tactical grid position.

        The waypoint grid is an 8x8 grid covering the battlefield with margins.

        Args:
            waypoint_index: Waypoint index (0 to waypoint_grid_size^2 - 1)

        Returns:
            (x, y) position in tactical grid coordinates
        """
        grid_x = waypoint_index % self.waypoint_grid_size
        grid_y = waypoint_index // self.waypoint_grid_size

        # Convert to actual grid position (with margins)
        margin = 2.0
        cell_size = (GRID_SIZE - 2 * margin) / self.waypoint_grid_size

        x = margin + (grid_x + 0.5) * cell_size
        y = margin + (grid_y + 0.5) * cell_size

        return (x, y)

    def position_to_waypoint_index(self, x: float, y: float) -> int:
        """
        Convert tactical grid position to waypoint index.

        Args:
            x: X position
            y: Y position

        Returns:
            Waypoint index
        """
        margin = 2.0
        cell_size = (GRID_SIZE - 2 * margin) / self.waypoint_grid_size

        grid_x = int((x - margin) / cell_size)
        grid_y = int((y - margin) / cell_size)

        # Clamp to valid range
        grid_x = max(0, min(self.waypoint_grid_size - 1, grid_x))
        grid_y = max(0, min(self.waypoint_grid_size - 1, grid_y))

        return grid_y * self.waypoint_grid_size + grid_x

    def get_all_waypoint_positions(self) -> List[Tuple[float, float]]:
        """
        Get list of all waypoint positions in the grid.

        Returns:
            List of (x, y) positions for each waypoint index
        """
        positions = []
        for i in range(self.waypoint_grid_size * self.waypoint_grid_size):
            positions.append(self.waypoint_index_to_position(i))
        return positions

    # ==================== Unit Queries ====================

    def get_unit_strategic_cell(self, unit_id: int, team: str) -> Optional[Tuple[int, int]]:
        """
        Get the strategic cell containing a unit's centroid.

        Args:
            unit_id: Unit ID
            team: Team ("blue" or "red")

        Returns:
            (cell_x, cell_y) or None if unit not found
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{team}_units', [])

        for unit in units:
            if unit.id == unit_id:
                centroid = unit.centroid
                return self.position_to_cell(centroid[0], centroid[1])

        return None

    def get_units_in_cell(self, cell_x: int, cell_y: int) -> Dict[str, List[int]]:
        """
        Get unit IDs in a strategic cell.

        Args:
            cell_x: Strategic cell X (0-3)
            cell_y: Strategic cell Y (0-3)

        Returns:
            Dict with 'blue' and 'red' lists of unit IDs
        """
        base_env = self.env.unwrapped
        blue_units = getattr(base_env, 'blue_units', [])
        red_units = getattr(base_env, 'red_units', [])

        result = {'blue': [], 'red': []}

        for unit in blue_units:
            centroid = unit.centroid
            unit_cell = self.position_to_cell(centroid[0], centroid[1])
            if unit_cell == (cell_x, cell_y):
                result['blue'].append(unit.id)

        for unit in red_units:
            centroid = unit.centroid
            unit_cell = self.position_to_cell(centroid[0], centroid[1])
            if unit_cell == (cell_x, cell_y):
                result['red'].append(unit.id)

        return result

    def get_agents_in_cell(self, cell_x: int, cell_y: int) -> Dict[str, int]:
        """
        Get count of agents in a strategic cell.

        Args:
            cell_x: Strategic cell X (0-3)
            cell_y: Strategic cell Y (0-3)

        Returns:
            Dict with 'blue' and 'red' agent counts
        """
        base_env = self.env.unwrapped
        blue_agents = getattr(base_env, 'blue_agents', [])
        red_agents = getattr(base_env, 'red_agents', [])

        x_start = cell_x * TACTICAL_CELLS_PER_STRATEGIC
        y_start = cell_y * TACTICAL_CELLS_PER_STRATEGIC
        x_end = x_start + TACTICAL_CELLS_PER_STRATEGIC
        y_end = y_start + TACTICAL_CELLS_PER_STRATEGIC

        blue_count = 0
        red_count = 0

        for agent in blue_agents:
            if agent.is_alive:
                x, y = agent.position
                if x_start <= x < x_end and y_start <= y < y_end:
                    blue_count += 1

        for agent in red_agents:
            if agent.is_alive:
                x, y = agent.position
                if x_start <= x < x_end and y_start <= y < y_end:
                    red_count += 1

        return {'blue': blue_count, 'red': red_count}

    # ==================== Unit-Level Observations ====================

    def get_unit_observation(self, unit_id: int, team: str) -> np.ndarray:
        """
        Get strategic observation for a single unit.

        Observation format (7 floats):
        - centroid_x, centroid_y (normalized)
        - average_health (0-1)
        - alive_ratio (0-1)
        - waypoint_x, waypoint_y (normalized, or 0.5 if none)
        - formation_state (0=cohesive, 0.5=scattered, 1=broken)

        Args:
            unit_id: Unit ID
            team: Team ("blue" or "red")

        Returns:
            numpy array of shape (7,)
        """
        base_env = self.env.unwrapped
        units = getattr(base_env, f'{team}_units', [])

        for unit in units:
            if unit.id == unit_id:
                return self._build_unit_observation(unit)

        return np.zeros(UNIT_OBS_SIZE, dtype=np.float32)

    def get_all_units_observation(self, team: Optional[str] = None) -> np.ndarray:
        """
        Get strategic observations for all units.

        Args:
            team: "blue", "red", or None for all

        Returns:
            numpy array of shape (N * 7,) where N is number of units
        """
        base_env = self.env.unwrapped
        observations = []

        if team is None or team == "blue":
            blue_units = getattr(base_env, 'blue_units', [])
            for unit in blue_units:
                observations.append(self._build_unit_observation(unit))

        if team is None or team == "red":
            red_units = getattr(base_env, 'red_units', [])
            for unit in red_units:
                observations.append(self._build_unit_observation(unit))

        if not observations:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(observations)

    def _build_unit_observation(self, unit) -> np.ndarray:
        """
        Build observation array for a single unit.

        Args:
            unit: Unit instance

        Returns:
            numpy array of shape (7,)
        """
        obs = np.zeros(UNIT_OBS_SIZE, dtype=np.float32)

        # Centroid position (normalized)
        centroid = unit.centroid
        obs[0] = centroid[0] / GRID_SIZE
        obs[1] = centroid[1] / GRID_SIZE

        # Average health
        alive_agents = unit.alive_agents
        if alive_agents:
            avg_health = sum(a.health for a in alive_agents) / len(alive_agents) / 100.0
        else:
            avg_health = 0.0
        obs[2] = avg_health

        # Alive ratio
        obs[3] = unit.alive_count / max(1, len(unit.agents))

        # Waypoint (or 0.5, 0.5 if none)
        if unit.waypoint is not None:
            obs[4] = unit.waypoint[0] / GRID_SIZE
            obs[5] = unit.waypoint[1] / GRID_SIZE
        else:
            obs[4] = 0.5
            obs[5] = 0.5

        # Formation state based on spread
        spread = unit.get_formation_spread()
        if spread < 2.0:
            obs[6] = 0.0  # Cohesive
        elif spread < 4.0:
            obs[6] = 0.5  # Scattered
        else:
            obs[6] = 1.0  # Broken

        return obs

    # ==================== Strategic Analysis ====================

    def get_contested_cells(self) -> List[Tuple[int, int]]:
        """
        Get cells where both teams have agents.

        Returns:
            List of (cell_x, cell_y) for contested cells
        """
        contested = []

        for cy in range(STRATEGIC_GRID_SIZE):
            for cx in range(STRATEGIC_GRID_SIZE):
                agents = self.get_agents_in_cell(cx, cy)
                if agents['blue'] > 0 and agents['red'] > 0:
                    contested.append((cx, cy))

        return contested

    def get_dominant_team_map(self) -> np.ndarray:
        """
        Get 4x4 map showing dominant team per cell.

        Values: -1 (blue), 0 (neutral/empty), 1 (red)

        Returns:
            numpy array of shape (4, 4)
        """
        occupancy = self.get_occupancy_board()

        # Convert 0.0/0.5/1.0 to -1/0/1
        dominant = np.zeros_like(occupancy, dtype=np.int8)
        dominant[occupancy < 0.25] = -1  # Blue
        dominant[occupancy > 0.75] = 1   # Red
        # Neutral stays 0

        return dominant

    def get_frontline_cells(self) -> List[Tuple[int, int]]:
        """
        Get cells on the frontline (adjacent to enemy-controlled cells).

        Returns:
            List of (cell_x, cell_y) for frontline cells
        """
        dominant = self.get_dominant_team_map()
        frontline = []

        for cy in range(STRATEGIC_GRID_SIZE):
            for cx in range(STRATEGIC_GRID_SIZE):
                current = dominant[cy, cx]
                if current == 0:  # Neutral
                    continue

                # Check adjacent cells
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < STRATEGIC_GRID_SIZE and 0 <= ny < STRATEGIC_GRID_SIZE:
                        neighbor = dominant[ny, nx]
                        if neighbor != 0 and neighbor != current:
                            frontline.append((cx, cy))
                            break

        return frontline

    def toggle_strategic_grid(self) -> bool:
        """
        Toggle strategic grid overlay visibility.

        Returns:
            New visibility state
        """
        self.show_strategic_grid = not self.show_strategic_grid
        self._sync_to_base_env()
        return self.show_strategic_grid
