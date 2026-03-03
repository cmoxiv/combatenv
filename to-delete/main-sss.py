"""
State Space Search Demo (main-sss.py)

Visualizes 5 classic search algorithms (BFS, DFS, Dijkstra, A*, Beam Search)
exploring the combatenv terrain grid side-by-side. Each algorithm expands one
node per step, showing frontier and visited cells in real-time. Once a path is
found, the agent walks it step-by-step while its combat stats deplete.

Dijkstra and A* use a combat-readiness reward as their edge cost, which
accounts for terrain speed penalties (delayed cost), stamina drain, and HP
loss.  BFS, DFS and Beam Search remain uninformed.

Controls:
    SPACE    Toggle pause / play
    RETURN   Skip -- finish all searches and path-walks instantly
    +/=      Speed up (more nodes per frame, max 20)
    -        Slow down (min 1)
    F        Toggle frontier/visited overlay
    I        Toggle info overlays (stats, legend, controls)
    0        Show/hide all agents
    1-5      Toggle visibility of each agent (DFS / BFS / Dijkstra / A* / Beam)
    R        Reset (new terrain, click new goal)
    Q        Quit
    Click    Set goal (on walkable cell)
"""

import heapq
import random
import sys
from abc import ABC, abstractmethod
from collections import deque
from copy import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from combatenv.gridworld import GridWorld
from combatenv.terrain import TerrainGrid, TerrainType
from combatenv.renderer import render_background, render_terrain, render_grid_lines
from combatenv import renderer as _renderer_module
from combatenv.config import GRID_SIZE, CELL_SIZE, WINDOW_SIZE, FPS

# ── Constants ───────────────────────────────────────────────────────────────

NOOP, NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3, 4
ACTION_DELTAS = {
    NOOP: (0, 0),
    NORTH: (0, -1),
    SOUTH: (0, 1),
    EAST: (1, 0),
    WEST: (-1, 0),
}
DELTA_TO_ACTION = {(0, -1): NORTH, (0, 1): SOUTH, (1, 0): EAST, (-1, 0): WEST}

# Simple terrain cost (used by BFS/DFS path-cost accounting)
TERRAIN_COST = {
    TerrainType.EMPTY: 1,
    TerrainType.FOREST: 2,
    TerrainType.WATER: 3,
    TerrainType.FIRE: 5,
    TerrainType.OBSTACLE: float("inf"),
}

# Per-terrain effects on agent stats
#   stamina_drain : base stamina lost per step
#   hp_drain      : HP lost per step
#   speed_mult    : time multiplier (>1 means slower, acts as "delayed penalty")
TERRAIN_EFFECTS: Dict[TerrainType, Dict[str, float]] = {
    TerrainType.EMPTY:    {"stamina_drain": 0.0,  "hp_drain": 0.0, "speed_mult": 1.0},
    TerrainType.FOREST:   {"stamina_drain": 0.0,  "hp_drain": 0.0, "speed_mult": 1.5},
    TerrainType.WATER:    {"stamina_drain": 5.0,  "hp_drain": 0.0, "speed_mult": 2.0},
    TerrainType.FIRE:     {"stamina_drain": 0.0,  "hp_drain": 15.0, "speed_mult": 1.0},
    TerrainType.OBSTACLE: {"stamina_drain": 0.0,  "hp_drain": 0.0, "speed_mult": float("inf")},
}

DROWNING_HP_DRAIN = 10.0     # extra HP/step when stamina=0 in water

# Weights for the combat-readiness cost function
STAMINA_COST_WEIGHT = 0.15
HP_COST_WEIGHT = 0.30

# Agent stat caps
MAX_HP = 100.0
MAX_STAMINA = 100.0
MAX_AMMO = 100
MAX_ARMOR = 100.0

# Visual
COLOR_BFS = (0, 120, 255)
COLOR_DFS = (160, 50, 220)
COLOR_DIJKSTRA = (200, 140, 0)
COLOR_ASTAR = (0, 180, 80)
COLOR_BEAM = (0, 180, 180)

BEAM_WIDTH = 32

VISITED_ALPHA = 40
FRONTIER_ALPHA = 80
PATH_ALPHA = 160

START = (1, 1)
SPAWN_CLEAR_RADIUS = 3

# Red team outposts (LOS-based corridor threats)
OUTPOST_FIRE_RANGE = 8       # Euclidean radius for danger zone
OUTPOST_FIRE_DAMAGE = 8.0    # HP lost per step in danger zone
NUM_OUTPOSTS = 4
COLOR_OUTPOST = (200, 0, 0)
OUTPOST_ZONE_ALPHA = 50      # semi-transparent red overlay
MIN_OUTPOST_DISTANCE = 15    # min Euclidean distance between outposts
OUTPOST_EDGE_MARGIN = 4      # min cells from map edge for outpost placement

# Healing springs
NUM_SPRINGS = 3
SPRING_HP_HEAL = 12.0        # HP restored per step on spring
SPRING_STAMINA_HEAL = 8.0    # stamina restored per step on spring
COLOR_SPRING = (40, 200, 80)
SPRING_ZONE_ALPHA = 40       # green overlay alpha
MIN_SPRING_DISTANCE = 12     # min Euclidean distance between springs
SPRING_HEAL_RANGE = 5        # cells — Euclidean radius for healing zone


# ── Agent Stats ─────────────────────────────────────────────────────────────

@dataclass
class AgentStats:
    """Mutable combat-readiness stats carried by each search agent."""
    hp: float = MAX_HP
    stamina: float = MAX_STAMINA
    ammo: int = MAX_AMMO
    armor: float = MAX_ARMOR

    @property
    def combat_readiness(self) -> float:
        """Normalised 0-1 readiness (weighted average favouring HP, zero if dead)."""
        if self.hp <= 0:
            return 0.0
        return (
            (self.hp / MAX_HP) * 0.50
            + (self.stamina / MAX_STAMINA) * 0.20
            + (self.ammo / MAX_AMMO) * 0.15
            + (self.armor / MAX_ARMOR) * 0.15
        )

    def apply_terrain(self, terrain: TerrainType) -> None:
        """Deplete stats for one step on *terrain*."""
        fx = TERRAIN_EFFECTS[terrain]
        # Stamina drain is scaled by speed_mult (delayed penalty:
        # slower terrain means more time spent, so more stamina consumed)
        effective_drain = fx["stamina_drain"] * fx["speed_mult"]
        self.stamina = max(0.0, self.stamina - effective_drain)

        # HP loss from terrain (fire)
        self.hp = max(0.0, self.hp - fx["hp_drain"])

        # Drowning: no stamina left while in water
        if terrain == TerrainType.WATER and self.stamina <= 0:
            self.hp = max(0.0, self.hp - DROWNING_HP_DRAIN)


# ── Search Agents ───────────────────────────────────────────────────────────

class SearchAgent(ABC):
    """Base class for all search algorithm agents.

    Agents communicate with the environment through a two-method protocol:
        get_actions() -> [(position, action), ...]   # agent says what it wants
        step(results) -> None                         # agent processes results

    The main loop mediates all environment interaction — agents never call
    env.step() directly.
    """

    def __init__(
        self,
        name: str,
        color: Tuple[int, int, int],
        start: Tuple[int, int],
        goal: Tuple[int, int],
        terrain_grid: TerrainGrid,
        danger_zone: Optional[Set[Tuple[int, int]]] = None,
        heal_zone: Optional[Set[Tuple[int, int]]] = None,
    ):
        self.name = name
        self.color = color
        self.start = start
        self.goal = goal
        self.terrain_grid = terrain_grid
        self.danger_zone: Set[Tuple[int, int]] = danger_zone if danger_zone is not None else set()
        self.heal_zone: Set[Tuple[int, int]] = heal_zone if heal_zone is not None else set()

        self.visited: Set[Tuple[int, int]] = set()
        self.visited_order: List[Tuple[int, int]] = []   # insertion order for incremental rendering
        self.came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        self.cost_so_far: Dict[Tuple[int, int], float] = {}
        self.path_found = False
        self.search_done = False
        self.planned_path: List[Tuple[int, int]] = []
        self.path_index = 0
        self.position = start
        self.nodes_expanded = 0
        self.path_cost = 0.0
        self.stats = AgentStats()

    # ── Public protocol ────────────────────────────────────────────────────

    def get_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        """Return a list of (position, action) pairs the agent wants executed."""
        if self.is_done:
            return []
        if self.search_done and self.path_found:
            return self._get_walk_action()
        if not self.search_done:
            return self._get_search_actions()
        return []

    def step(self, results: List[Tuple[float, dict]]) -> None:
        """Process the results of the actions returned by get_actions()."""
        if self.is_done:
            return
        if self.search_done and self.path_found:
            self._process_walk(results)
        elif not self.search_done:
            self._process_search(results)

    @property
    def is_done(self) -> bool:
        """True when search is complete and walk is finished (or no path)."""
        if not self.search_done:
            return False
        if self.path_found:
            return self.path_index >= len(self.planned_path)
        return True  # search done, no path found

    # ── Walk phase ─────────────────────────────────────────────────────────

    def _get_walk_action(self) -> List[Tuple[Tuple[int, int], int]]:
        """Compute one step along the planned path."""
        if self.path_index >= len(self.planned_path):
            return []
        target = self.planned_path[self.path_index]
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        action = DELTA_TO_ACTION.get((dx, dy), NOOP)
        return [(self.position, action)]

    def _process_walk(self, results: List[Tuple[float, dict]]) -> None:
        """Advance one step along the path using env results."""
        if not results:
            return
        _, info = results[0]
        if info.get("moved", False):
            self.position = info["position"]
        self._apply_terrain_effects(self.position)
        # Outpost danger zone HP damage
        if self.position in self.danger_zone:
            self.stats.hp = max(0.0, self.stats.hp - OUTPOST_FIRE_DAMAGE)
        # Healing spring restoration (within heal zone radius)
        if self.position in self.heal_zone:
            self.stats.hp = min(MAX_HP, self.stats.hp + SPRING_HP_HEAL)
            self.stats.stamina = min(MAX_STAMINA, self.stats.stamina + SPRING_STAMINA_HEAL)
        self.path_index += 1

    # ── Search phase (subclass responsibility) ─────────────────────────────

    @abstractmethod
    def _get_search_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        """Return env probes needed for one search expansion ([] if none)."""
        pass

    @abstractmethod
    def _process_search(self, results: List[Tuple[float, dict]]) -> None:
        """Process probe results and update internal search state."""
        pass

    # ── Shared helpers ─────────────────────────────────────────────────────

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        neighbors = []
        for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.terrain_grid.is_walkable(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def _terrain_cost(self, node: Tuple[int, int]) -> float:
        """Simple terrain cost (used by BFS/DFS for internal bookkeeping)."""
        terrain = self.terrain_grid.get(node[0], node[1])
        return TERRAIN_COST.get(terrain, 1)

    def _combat_readiness_cost(self, pos: Tuple[int, int], terrain: TerrainType) -> float:
        """Same formula as CombatReadinessRewardWrapper, plus outpost/spring effects."""
        fx = TERRAIN_EFFECTS[terrain]
        cost = (fx["speed_mult"]
                + STAMINA_COST_WEIGHT * fx["stamina_drain"] * fx["speed_mult"]
                + HP_COST_WEIGHT * fx["hp_drain"])
        if pos in self.danger_zone:
            cost += HP_COST_WEIGHT * OUTPOST_FIRE_DAMAGE
        if pos in self.heal_zone:
            cost -= HP_COST_WEIGHT * SPRING_HP_HEAL
        return max(cost, 0.01)  # Dijkstra/A* require non-negative edge costs

    def _uniform_path_cost(self, path: List[Tuple[int, int]]) -> float:
        """Compute path cost using the combat-readiness cost for all agents."""
        return sum(
            self._combat_readiness_cost((x, y), self.terrain_grid.get(x, y))
            for x, y in path[1:]  # skip start cell
        )

    def _reconstruct_path(self) -> None:
        path: List[Tuple[int, int]] = []
        current: Optional[Tuple[int, int]] = self.goal
        seen: Set[Tuple[int, int]] = set()
        while current is not None and current != self.start:
            if current in seen:
                # Cycle in came_from (negative edge costs) — abort
                self.path_found = False
                return
            seen.add(current)
            path.append(current)
            current = self.came_from.get(current)
        path.append(self.start)
        path.reverse()
        self.planned_path = path
        self.path_index = 1  # skip start cell; agent is already there
        self.path_cost = self._uniform_path_cost(path)

    def _apply_terrain_effects(self, pos: Tuple[int, int]) -> None:
        terrain = self.terrain_grid.get(pos[0], pos[1])
        self.stats.apply_terrain(terrain)

    @abstractmethod
    def frontier_as_set(self) -> Set[Tuple[int, int]]:
        pass

    @property
    def walking_done(self) -> bool:
        return self.path_index >= len(self.planned_path)


class BFSAgent(SearchAgent):
    """Breadth-first search -- FIFO frontier, uniform cost."""

    def __init__(self, start, goal, terrain_grid, **kw):
        super().__init__("BFS", COLOR_BFS, start, goal, terrain_grid, **kw)
        self.frontier: deque = deque()
        self.frontier.append(self.start)
        self.visited.add(self.start)
        self.visited_order.append(self.start)
        self.came_from[self.start] = None

    def _get_search_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        return []  # BFS needs no env interaction

    def _process_search(self, results: List[Tuple[float, dict]]) -> None:
        self._expand_one_node()

    def _expand_one_node(self) -> None:
        if not self.frontier:
            self.search_done = True
            return
        node = self.frontier.popleft()
        self.nodes_expanded += 1
        if node == self.goal:
            self.path_found = True
            self.search_done = True
            self._reconstruct_path()
            return
        for neighbor in self._get_neighbors(node):
            if neighbor not in self.visited:
                self.visited.add(neighbor)
                self.visited_order.append(neighbor)
                self.came_from[neighbor] = node
                self.frontier.append(neighbor)

    def frontier_as_set(self) -> Set[Tuple[int, int]]:
        return set(self.frontier)


class DFSAgent(SearchAgent):
    """Depth-first search -- LIFO frontier (stack), uniform cost."""

    def __init__(self, start, goal, terrain_grid, **kw):
        super().__init__("DFS", COLOR_DFS, start, goal, terrain_grid, **kw)
        self.frontier: list = [self.start]
        self.visited.add(self.start)
        self.visited_order.append(self.start)
        self.came_from[self.start] = None

    def _get_search_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        return []  # DFS needs no env interaction

    def _process_search(self, results: List[Tuple[float, dict]]) -> None:
        self._expand_one_node()

    def _expand_one_node(self) -> None:
        if not self.frontier:
            self.search_done = True
            return
        node = self.frontier.pop()
        self.nodes_expanded += 1
        if node == self.goal:
            self.path_found = True
            self.search_done = True
            self._reconstruct_path()
            return
        for neighbor in self._get_neighbors(node):
            if neighbor not in self.visited:
                self.visited.add(neighbor)
                self.visited_order.append(neighbor)
                self.came_from[neighbor] = node
                self.frontier.append(neighbor)

    def frontier_as_set(self) -> Set[Tuple[int, int]]:
        return set(self.frontier)


class DijkstraAgent(SearchAgent):
    """Dijkstra -- priority queue by cumulative cost (uses reward wrapper)."""

    def __init__(self, start, goal, terrain_grid, **kw):
        super().__init__("Dijkstra", COLOR_DIJKSTRA, start, goal, terrain_grid, **kw)
        self._counter = 0
        self.frontier: list = []
        self.cost_so_far[self.start] = 0
        heapq.heappush(self.frontier, (0, self._counter, self.start))
        self._counter += 1
        self.came_from[self.start] = None
        self._expanding_node: Optional[Tuple[int, int]] = None

    def _get_search_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        # Pop nodes until we find one not yet visited
        while self.frontier:
            _, _, node = heapq.heappop(self.frontier)
            if node not in self.visited:
                break
        else:
            self.search_done = True
            return []

        self.visited.add(node)
        self.visited_order.append(node)
        self.nodes_expanded += 1

        if node == self.goal:
            self.path_found = True
            self.search_done = True
            self._reconstruct_path()
            return []

        # Request probes in all 4 directions from this node
        self._expanding_node = node
        return [(node, action) for action in (NORTH, SOUTH, EAST, WEST)]

    def _process_search(self, results: List[Tuple[float, dict]]) -> None:
        node = self._expanding_node
        if node is None:
            return
        for reward, info in results:
            if not info["moved"]:
                continue
            neighbor = info["position"]
            edge_cost = -reward
            new_cost = self.cost_so_far[node] + edge_cost
            if neighbor not in self.cost_so_far or new_cost < self.cost_so_far[neighbor]:
                self.cost_so_far[neighbor] = new_cost
                self.came_from[neighbor] = node
                heapq.heappush(self.frontier, (new_cost, self._counter, neighbor))
                self._counter += 1
        self._expanding_node = None

    def frontier_as_set(self) -> Set[Tuple[int, int]]:
        # Cap iteration to avoid O(n) on huge heaps
        result: Set[Tuple[int, int]] = set()
        for _, _, node in self.frontier:
            if node not in self.visited:
                result.add(node)
            if len(result) >= 200:
                break
        return result


class AStarAgent(SearchAgent):
    """A* -- priority queue by f = g + Manhattan heuristic (uses reward wrapper)."""

    def __init__(self, start, goal, terrain_grid, **kw):
        super().__init__("A*", COLOR_ASTAR, start, goal, terrain_grid, **kw)
        self._counter = 0
        self.frontier: list = []
        self.cost_so_far[self.start] = 0
        f = self._heuristic(self.start)
        heapq.heappush(self.frontier, (f, self._counter, self.start))
        self._counter += 1
        self.came_from[self.start] = None
        self._expanding_node: Optional[Tuple[int, int]] = None

    def _heuristic(self, node: Tuple[int, int]) -> float:
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def _get_search_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        while self.frontier:
            _, _, node = heapq.heappop(self.frontier)
            if node not in self.visited:
                break
        else:
            self.search_done = True
            return []

        self.visited.add(node)
        self.visited_order.append(node)
        self.nodes_expanded += 1

        if node == self.goal:
            self.path_found = True
            self.search_done = True
            self._reconstruct_path()
            return []

        self._expanding_node = node
        return [(node, action) for action in (NORTH, SOUTH, EAST, WEST)]

    def _process_search(self, results: List[Tuple[float, dict]]) -> None:
        node = self._expanding_node
        if node is None:
            return
        for reward, info in results:
            if not info["moved"]:
                continue
            neighbor = info["position"]
            edge_cost = -reward
            new_cost = self.cost_so_far[node] + edge_cost
            if neighbor not in self.cost_so_far or new_cost < self.cost_so_far[neighbor]:
                self.cost_so_far[neighbor] = new_cost
                self.came_from[neighbor] = node
                f = new_cost + self._heuristic(neighbor)
                heapq.heappush(self.frontier, (f, self._counter, neighbor))
                self._counter += 1
        self._expanding_node = None

    def frontier_as_set(self) -> Set[Tuple[int, int]]:
        # Cap iteration to avoid O(n) on huge heaps
        result: Set[Tuple[int, int]] = set()
        for _, _, node in self.frontier:
            if node not in self.visited:
                result.add(node)
            if len(result) >= 200:
                break
        return result


class BeamSearchAgent(SearchAgent):
    """Beam Search -- bounded-width BFS using Manhattan distance heuristic.

    Expands level-by-level like BFS, but at each level keeps only the top-k
    nodes (lowest heuristic = closest to goal). Uninformed: no env probes.
    """

    def __init__(self, start, goal, terrain_grid, **kw):
        super().__init__("Beam", COLOR_BEAM, start, goal, terrain_grid, **kw)
        self.current_beam: List[Tuple[int, int]] = [start]
        self.next_candidates: List[Tuple[int, int]] = []
        self.beam_index: int = 0
        self.visited.add(start)
        self.visited_order.append(start)
        self.came_from[start] = None

    def _heuristic(self, node: Tuple[int, int]) -> float:
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def _get_search_actions(self) -> List[Tuple[Tuple[int, int], int]]:
        return []

    def _process_search(self, results: List[Tuple[float, dict]]) -> None:
        self._expand_one_node()

    def _expand_one_node(self) -> None:
        # If current beam exhausted, form next beam from candidates
        if self.beam_index >= len(self.current_beam):
            # Dedup candidates against visited
            unique = [c for c in self.next_candidates if c not in self.visited]
            if not unique:
                self.search_done = True
                return
            # Sort by heuristic, keep top BEAM_WIDTH
            unique.sort(key=self._heuristic)
            self.current_beam = unique[:BEAM_WIDTH]
            # Mark new beam members as visited
            for node in self.current_beam:
                self.visited.add(node)
                self.visited_order.append(node)
            self.beam_index = 0
            self.next_candidates = []

        node = self.current_beam[self.beam_index]
        self.beam_index += 1
        self.nodes_expanded += 1

        if node == self.goal:
            self.path_found = True
            self.search_done = True
            self._reconstruct_path()
            return

        for neighbor in self._get_neighbors(node):
            if neighbor not in self.visited:
                if neighbor not in self.came_from:
                    self.came_from[neighbor] = node
                self.next_candidates.append(neighbor)

    def frontier_as_set(self) -> Set[Tuple[int, int]]:
        return set(self.current_beam[self.beam_index:])


# ── Gymnasium Wrappers ──────────────────────────────────────────────────────

class FullyObservableWrapper(gym.Wrapper):
    """Observation = terrain grid as int8 numpy array (cached)."""

    def __init__(self, env: GridWorld, terrain_grid: TerrainGrid):
        super().__init__(env)
        self.terrain_grid = terrain_grid
        self.observation_space = spaces.Box(0, 4, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self._cached_obs: Optional[np.ndarray] = None

    def _get_obs(self) -> np.ndarray:
        if self._cached_obs is None:
            obs = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    obs[x, y] = self.terrain_grid.get(x, y)
            self._cached_obs = obs
        return self._cached_obs

    def invalidate_obs(self) -> None:
        self._cached_obs = None

    def reset(self, **kwargs):
        self._cached_obs = None
        _, info = self.env.reset(**kwargs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._get_obs(), reward, term, trunc, info


class CardinalActionWrapper(gym.Wrapper):
    """Discrete(5) cardinal movement with position tracking."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(5)
        self.position: Tuple[int, int] = START

    def set_position(self, pos: Tuple[int, int]) -> None:
        self.position = pos

    def get_position(self) -> Tuple[int, int]:
        return self.position

    def step(self, action):
        dx, dy = ACTION_DELTAS[action]
        nx, ny = self.position[0] + dx, self.position[1] + dy
        terrain_grid = self.env.terrain_grid  # from FullyObservableWrapper
        moved = False
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and terrain_grid.is_walkable(nx, ny):
            self.position = (nx, ny)
            moved = True
        obs, _, term, trunc, info = self.env.step(action)
        info["position"] = self.position
        info["moved"] = moved
        return obs, 0.0, term, trunc, info  # base reward = 0; reward wrapper adds cost


class CombatReadinessRewardWrapper(gym.Wrapper):
    """Computes reward from transition cost via env.step().

    Cost = speed_mult                                       (time / delayed penalty)
         + STAMINA_COST_WEIGHT * stamina_drain * speed_mult (stamina lost over time)
         + HP_COST_WEIGHT      * hp_drain                   (direct HP damage)

    Informed search agents (Dijkstra, A*) call env.step(action) and read
    -reward as the edge cost.
    """

    def __init__(self, env, terrain_grid: TerrainGrid,
                 danger_zone: Optional[Set[Tuple[int, int]]] = None,
                 heal_zone: Optional[Set[Tuple[int, int]]] = None):
        super().__init__(env)
        self.terrain_grid = terrain_grid
        # Store a reference to a mutable set; on reset the caller clears/repopulates it.
        self.danger_zone: Set[Tuple[int, int]] = danger_zone if danger_zone is not None else set()
        self.heal_zone: Set[Tuple[int, int]] = heal_zone if heal_zone is not None else set()

    def set_position(self, pos: Tuple[int, int]) -> None:
        self.env.set_position(pos)

    def get_position(self) -> Tuple[int, int]:
        return self.env.get_position()

    def step(self, action):
        obs, _, term, trunc, info = self.env.step(action)
        pos = info["position"]
        moved = info["moved"]
        if not moved:
            reward = 0.0
        else:
            terrain = self.terrain_grid.get(pos[0], pos[1])
            fx = TERRAIN_EFFECTS[terrain]
            cost = (fx["speed_mult"]
                    + STAMINA_COST_WEIGHT * fx["stamina_drain"] * fx["speed_mult"]
                    + HP_COST_WEIGHT * fx["hp_drain"])
            # Outpost danger zone penalty
            if pos in self.danger_zone:
                cost += HP_COST_WEIGHT * OUTPOST_FIRE_DAMAGE
            # Healing spring benefit
            if pos in self.heal_zone:
                cost -= HP_COST_WEIGHT * SPRING_HP_HEAL
            cost = max(cost, 0.01)  # Dijkstra/A* require non-negative edge costs
            reward = -cost
        info["combat_readiness"] = {
            "hp": MAX_HP,
            "stamina": MAX_STAMINA,
            "ammo": MAX_AMMO,
            "armor": MAX_ARMOR,
        }
        return obs, reward, term, trunc, info


class SearchVisualiserWrapper(gym.Wrapper):
    """Stores references to SearchAgents and renders their overlays."""

    def __init__(self, env, agents: List[SearchAgent]):
        super().__init__(env)
        self.agents = agents
        self._surfaces_ready = False
        self._overlays: Dict[str, Dict[str, pygame.Surface]] = {}

    def _ensure_surfaces(self) -> None:
        if self._surfaces_ready:
            return
        for agent in self.agents:
            r, g, b = agent.color
            visited = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            visited.fill((r, g, b, VISITED_ALPHA))
            frontier = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            frontier.fill((r, g, b, FRONTIER_ALPHA))
            pygame.draw.rect(frontier, (r, g, b, min(255, FRONTIER_ALPHA + 100)), (0, 0, CELL_SIZE, CELL_SIZE), 1)
            path = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            path.fill((r, g, b, PATH_ALPHA))
            self._overlays[agent.name] = {
                "visited": visited,
                "frontier": frontier,
                "path": path,
            }
        self._surfaces_ready = True

    def _build_final_cache(self, agent: SearchAgent) -> None:
        """Finalize visited + path overlay surfaces when search completes."""
        self._ensure_surfaces()
        surfs = self._overlays.get(agent.name)
        if surfs is None:
            return
        # Flush any remaining visited cells onto the running surface
        live = surfs.get("_live_surf")
        if live is None:
            live = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        tile = surfs["visited"]
        drawn = surfs.get("_drawn_count", 0)
        for cell in agent.visited_order[drawn:]:
            live.blit(tile, (cell[0] * CELL_SIZE, cell[1] * CELL_SIZE))
        surfs["_visited_cache"] = live.convert_alpha()
        # Path surface
        if agent.path_found and agent.planned_path:
            path_surf = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            ptile = surfs["path"]
            for x, y in agent.planned_path:
                path_surf.blit(ptile, (x * CELL_SIZE, y * CELL_SIZE))
            surfs["_path_cache"] = path_surf.convert_alpha()
        # Clean up live rendering state
        surfs.pop("_live_surf", None)
        surfs.pop("_drawn_count", None)
        # Free search data structures no longer needed
        if hasattr(agent, 'frontier'):
            agent.frontier.clear()
        if hasattr(agent, 'current_beam'):
            agent.current_beam.clear()
            agent.next_candidates.clear()
        agent.came_from.clear()
        agent.cost_so_far.clear()

    def render_search_overlays(
        self,
        screen: pygame.Surface,
        visible: List[bool],
        show_visited: bool = True,
    ) -> None:
        self._ensure_surfaces()
        for i, agent in enumerate(self.agents):
            if not visible[i]:
                continue
            surfs = self._overlays[agent.name]
            hide = not show_visited and agent.search_done

            if agent.search_done:
                # ── Post-search: use final caches ──
                if not hide:
                    cache = surfs.get("_visited_cache")
                    if cache is not None:
                        screen.blit(cache, (0, 0))
                if agent.path_found:
                    path_cache = surfs.get("_path_cache")
                    if path_cache is not None:
                        screen.blit(path_cache, (0, 0))
                    else:
                        ptile = surfs["path"]
                        for x, y in agent.planned_path:
                            screen.blit(ptile, (x * CELL_SIZE, y * CELL_SIZE))
            else:
                # ── During search: incremental live surface ──
                live = surfs.get("_live_surf")
                if live is None:
                    live = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
                    surfs["_live_surf"] = live
                    surfs["_drawn_count"] = 0

                # Draw only NEW visited cells since last frame
                drawn = surfs["_drawn_count"]
                total = len(agent.visited_order)
                if total > drawn:
                    tile = surfs["visited"]
                    for cell in agent.visited_order[drawn:total]:
                        live.blit(tile, (cell[0] * CELL_SIZE, cell[1] * CELL_SIZE))
                    surfs["_drawn_count"] = total

                if not hide:
                    screen.blit(live, (0, 0))
                    # Draw frontier (small set, changes each frame)
                    ftile = surfs["frontier"]
                    for x, y in agent.frontier_as_set():
                        screen.blit(ftile, (x * CELL_SIZE, y * CELL_SIZE))

    def invalidate_surfaces(self) -> None:
        self._surfaces_ready = False
        self._overlays.clear()


# ── Application State ───────────────────────────────────────────────────────

@dataclass
class AppState:
    """Mutable state bag for the main loop."""
    running: bool = True
    paused: bool = True          # starts paused, waiting for goal click
    speed: int = 4
    goal: Optional[Tuple[int, int]] = None
    waiting_for_goal: bool = True
    skip_requested: bool = False
    visible: List[bool] = field(default_factory=lambda: [True] * 5)
    show_visited: bool = True    # F key toggles visited cells overlay
    show_info: bool = True       # I key toggles info overlays (stats, legend, controls)
    outpost_positions: Set[Tuple[int, int]] = field(default_factory=set)
    danger_zone: Set[Tuple[int, int]] = field(default_factory=set)
    spring_positions: Set[Tuple[int, int]] = field(default_factory=set)
    heal_zone: Set[Tuple[int, int]] = field(default_factory=set)


# ── Event Handling ──────────────────────────────────────────────────────────

def handle_events(
    state: AppState,
    agents: List[SearchAgent],
    terrain_grid: TerrainGrid,
    vis_env: SearchVisualiserWrapper,
    make_agents,
) -> Optional[List[SearchAgent]]:
    """Process all pygame events, mutate *state* in-place.

    Returns a new agent list when a reset or new-goal creates fresh agents,
    otherwise returns None (caller keeps current list).
    """
    new_agents: Optional[List[SearchAgent]] = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            state.running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                state.running = False

            elif event.key == pygame.K_SPACE:
                if not state.waiting_for_goal:
                    state.paused = not state.paused

            elif event.key == pygame.K_RETURN:
                if not state.waiting_for_goal:
                    state.skip_requested = True

            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                state.speed = min(20, state.speed + 1)

            elif event.key == pygame.K_MINUS:
                state.speed = max(1, state.speed - 1)

            elif event.key == pygame.K_f:
                state.show_visited = not state.show_visited

            elif event.key == pygame.K_i:
                state.show_info = not state.show_info

            elif event.key == pygame.K_r:
                _renderer_module._TERRAIN_SURFACE = None
                generate_terrain(terrain_grid)
                # Regenerate outposts — clear + repopulate the existing set objects
                # so the reward wrapper's reference stays valid.
                new_outposts, new_danger = _place_outposts(terrain_grid)
                state.outpost_positions.clear()
                state.outpost_positions.update(new_outposts)
                state.danger_zone.clear()
                state.danger_zone.update(new_danger)
                # Regenerate healing springs
                new_springs, new_heal = _place_springs(terrain_grid, state.outpost_positions)
                state.spring_positions.clear()
                state.spring_positions.update(new_springs)
                state.heal_zone.clear()
                state.heal_zone.update(new_heal)
                _invalidate_overlay_caches()
                state.goal = None
                state.waiting_for_goal = True
                state.paused = True
                new_agents = make_agents(None)
                vis_env.agents = new_agents
                vis_env.invalidate_surfaces()

            elif event.key == pygame.K_1:
                state.visible[0] = not state.visible[0]
            elif event.key == pygame.K_2:
                state.visible[1] = not state.visible[1]
            elif event.key == pygame.K_3:
                state.visible[2] = not state.visible[2]
            elif event.key == pygame.K_4:
                state.visible[3] = not state.visible[3]
            elif event.key == pygame.K_5:
                state.visible[4] = not state.visible[4]
            elif event.key == pygame.K_0:
                new_val = not all(state.visible)
                state.visible = [new_val] * len(state.visible)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            gx = mx // CELL_SIZE
            gy = my // CELL_SIZE
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE and terrain_grid.is_walkable(gx, gy):
                all_done = agents and all(a.is_done for a in agents)
                if state.waiting_for_goal or all_done:
                    state.goal = (gx, gy)
                    state.waiting_for_goal = False
                    state.paused = False
                    if all_done:
                        # Chain: agents continue from current positions
                        new_agents = make_agents(state.goal, agents)
                    else:
                        new_agents = make_agents(state.goal)
                    vis_env.agents = new_agents
                    vis_env.invalidate_surfaces()

    return new_agents


# ── Rendering Helpers ───────────────────────────────────────────────────────

def render_goal(screen: pygame.Surface, goal: Tuple[int, int]) -> None:
    gx = goal[0] * CELL_SIZE + CELL_SIZE // 2
    gy = goal[1] * CELL_SIZE + CELL_SIZE // 2
    size = CELL_SIZE // 2 + 2
    gold = (255, 215, 0)
    border = (180, 150, 0)
    points = [(gx, gy - size), (gx + size, gy), (gx, gy + size), (gx - size, gy)]
    pygame.draw.polygon(screen, gold, points)
    pygame.draw.polygon(screen, border, points, 2)


def render_start_marker(screen: pygame.Surface) -> None:
    sx = START[0] * CELL_SIZE + CELL_SIZE // 2
    sy = START[1] * CELL_SIZE + CELL_SIZE // 2
    size = CELL_SIZE // 2 + 2
    pygame.draw.rect(
        screen, (100, 255, 100),
        (sx - size, sy - size, size * 2, size * 2), 2,
    )


def render_agent_markers(
    screen: pygame.Surface,
    agents: List[SearchAgent],
    visible: List[bool],
) -> None:
    for i, agent in enumerate(agents):
        if not visible[i]:
            continue
        px = agent.position[0] * CELL_SIZE + CELL_SIZE // 2
        py = agent.position[1] * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2 - 1
        pygame.draw.circle(screen, agent.color, (px, py), radius)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), radius, 2)


def render_waiting_prompt(screen: pygame.Surface) -> None:
    font = pygame.font.Font(None, 36)
    text = font.render("Click a walkable cell to set the goal", True, (255, 255, 255))
    rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    bg = pygame.Surface((rect.width + 30, rect.height + 20), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 180))
    screen.blit(bg, (rect.x - 15, rect.y - 10))
    screen.blit(text, rect)


def render_next_goal_prompt(screen: pygame.Surface) -> None:
    font = pygame.font.Font(None, 30)
    text = font.render("Click a walkable cell to set the next goal", True, (200, 255, 200))
    rect = text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE - 40))
    bg = pygame.Surface((rect.width + 20, rect.height + 14), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    screen.blit(bg, (rect.x - 10, rect.y - 7))
    screen.blit(text, rect)


def _readiness_color(cr: float) -> Tuple[int, int, int]:
    """Green at 1.0, yellow at 0.5, red at 0.0."""
    if cr >= 0.5:
        t = (cr - 0.5) * 2.0  # 0..1
        return (int(255 * (1 - t)), 255, 0)
    t = cr * 2.0  # 0..1
    return (255, int(255 * t), 0)


def _draw_bar(
    surface: pygame.Surface,
    x: int, y: int, w: int, h: int,
    fraction: float,
    fill_color: Tuple[int, int, int],
) -> None:
    """Draw a tiny stat bar (background + filled portion)."""
    pygame.draw.rect(surface, (60, 60, 60), (x, y, w, h))
    fw = max(0, int(w * min(1.0, fraction)))
    if fw > 0:
        pygame.draw.rect(surface, fill_color, (x, y, fw, h))
    pygame.draw.rect(surface, (120, 120, 120), (x, y, w, h), 1)


def render_stats_overlay(
    screen: pygame.Surface,
    agents: List[SearchAgent],
    visible: List[bool],
    speed: int,
    paused: bool,
) -> None:
    panel_w = 260
    panel_h = 56 + len(agents) * 116
    panel_x = WINDOW_SIZE - panel_w - 10
    panel_y = 10

    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))
    pygame.draw.rect(panel, (200, 200, 200), (0, 0, panel_w, panel_h), 1)

    title_font = pygame.font.Font(None, 26)
    label_font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 19)
    tiny_font = pygame.font.Font(None, 17)

    y = 10
    panel.blit(title_font.render("SEARCH ALGORITHMS", True, (255, 255, 255)), (10, y))
    y += 28

    status_str = "PAUSED" if paused else f"Speed: {speed}x"
    status_color = (255, 100, 100) if paused else (100, 255, 100)
    panel.blit(label_font.render(status_str, True, status_color), (10, y))
    y += 24

    bar_w = 70
    bar_h = 8

    for i, agent in enumerate(agents):
        alpha_color = agent.color if visible[i] else tuple(c // 3 for c in agent.color)

        pygame.draw.rect(panel, alpha_color, (10, y + 2, 12, 12))
        if not visible[i]:
            pygame.draw.line(panel, (200, 200, 200), (10, y + 2), (22, y + 14), 1)
        panel.blit(label_font.render(agent.name, True, alpha_color), (28, y))
        y += 20

        # Status line
        if agent.path_found and agent.walking_done:
            s_text, s_col = "DONE", (100, 255, 100)
        elif agent.path_found:
            s_text = f"Walking ({agent.path_index}/{len(agent.planned_path)})"
            s_col = (200, 200, 100)
        elif agent.search_done:
            s_text, s_col = "No path", (255, 100, 100)
        else:
            s_text, s_col = "Searching...", (200, 200, 200)
        panel.blit(small_font.render(f"  {s_text}", True, s_col), (10, y))
        y += 16

        # Expanded / frontier
        panel.blit(small_font.render(f"  Expanded: {agent.nodes_expanded}", True, (180, 180, 180)), (10, y))
        y += 16

        f_size = len(agent.frontier_as_set()) if not agent.search_done else 0
        panel.blit(small_font.render(f"  Frontier: {f_size}", True, (180, 180, 180)), (10, y))
        y += 16

        # Path info
        if agent.path_found:
            panel.blit(
                small_font.render(
                    f"  Path: {len(agent.planned_path)} cells  cost {agent.path_cost:.1f}",
                    True, (180, 180, 180),
                ),
                (10, y),
            )
        y += 16

        # ── Combat readiness bars ──
        st = agent.stats
        cr = st.combat_readiness
        cr_col = _readiness_color(cr)

        # HP bar
        panel.blit(tiny_font.render("HP", True, (180, 180, 180)), (14, y))
        _draw_bar(panel, 34, y, bar_w, bar_h, st.hp / MAX_HP, (220, 60, 60))
        # Stamina bar
        panel.blit(tiny_font.render("STA", True, (180, 180, 180)), (112, y))
        _draw_bar(panel, 138, y, bar_w, bar_h, st.stamina / MAX_STAMINA, (60, 180, 220))
        y += 14

        # Readiness number
        panel.blit(
            tiny_font.render(f"  Readiness: {cr:.0%}", True, cr_col),
            (10, y),
        )
        y += 18

    screen.blit(panel, (panel_x, panel_y))


def render_legend(
    screen: pygame.Surface,
    agents: List[SearchAgent],
    visible: List[bool],
) -> None:
    font = pygame.font.Font(None, 22)
    block_h = 25 * len(agents) + 10
    x, y = 10, WINDOW_SIZE - block_h - 10

    bg = pygame.Surface((200, block_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 150))
    screen.blit(bg, (x, y))

    for i, agent in enumerate(agents):
        on = visible[i]
        color = agent.color if on else tuple(c // 3 for c in agent.color)
        label = f"{i + 1}: {agent.name}" + ("" if on else "  [hidden]")
        text_color = (220, 220, 220) if on else (120, 120, 120)

        pygame.draw.rect(screen, color, (x + 8, y + 6, 14, 14))
        if not on:
            pygame.draw.line(screen, (200, 200, 200), (x + 8, y + 6), (x + 22, y + 20), 1)
        screen.blit(font.render(label, True, text_color), (x + 28, y + 5))
        y += 25


_DANGER_ZONE_SURFACE: Optional[pygame.Surface] = None
_DANGER_ZONE_DIRTY = True


def _invalidate_overlay_caches() -> None:
    """Mark all static overlay caches as dirty (call after terrain/outpost/spring reset)."""
    global _DANGER_ZONE_DIRTY, _SPRINGS_DIRTY
    _DANGER_ZONE_DIRTY = True
    _SPRINGS_DIRTY = True


def render_danger_zone(screen: pygame.Surface, danger_zone: Set[Tuple[int, int]]) -> None:
    """Semi-transparent red overlay on all danger-zone cells (cached)."""
    global _DANGER_ZONE_SURFACE, _DANGER_ZONE_DIRTY
    if not danger_zone:
        return
    if _DANGER_ZONE_SURFACE is not None and not _DANGER_ZONE_DIRTY:
        screen.blit(_DANGER_ZONE_SURFACE, (0, 0))
        return
    _DANGER_ZONE_SURFACE = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    tile = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
    tile.fill((200, 0, 0, OUTPOST_ZONE_ALPHA))
    for x, y in danger_zone:
        _DANGER_ZONE_SURFACE.blit(tile, (x * CELL_SIZE, y * CELL_SIZE))
    _DANGER_ZONE_DIRTY = False
    screen.blit(_DANGER_ZONE_SURFACE, (0, 0))


def render_outposts(screen: pygame.Surface, outpost_positions: Set[Tuple[int, int]]) -> None:
    """Red circle with dark-red crosshair lines at each outpost."""
    for cx, cy in outpost_positions:
        px = cx * CELL_SIZE + CELL_SIZE // 2
        py = cy * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 2
        pygame.draw.circle(screen, COLOR_OUTPOST, (px, py), radius)
        pygame.draw.circle(screen, (120, 0, 0), (px, py), radius, 2)
        # Crosshair lines
        arm = radius + 2
        dark = (120, 0, 0)
        pygame.draw.line(screen, dark, (px - arm, py), (px + arm, py), 2)
        pygame.draw.line(screen, dark, (px, py - arm), (px, py + arm), 2)


def render_outpost_fire(
    screen: pygame.Surface,
    outpost_positions: Set[Tuple[int, int]],
    agents: List[SearchAgent],
    visible: List[bool],
    terrain_grid: TerrainGrid,
) -> None:
    """Draw red firing lines from outposts to walking agents in danger zones."""
    for i, agent in enumerate(agents):
        if not visible[i]:
            continue
        if not agent.path_found or agent.walking_done:
            continue
        ax, ay = agent.position
        for ox, oy in outpost_positions:
            dx = ax - ox
            dy = ay - oy
            if dx * dx + dy * dy > OUTPOST_FIRE_RANGE * OUTPOST_FIRE_RANGE:
                continue
            if not _has_los(terrain_grid, ox, oy, ax, ay):
                continue
            # Draw firing line
            sx = ox * CELL_SIZE + CELL_SIZE // 2
            sy = oy * CELL_SIZE + CELL_SIZE // 2
            ex = ax * CELL_SIZE + CELL_SIZE // 2
            ey = ay * CELL_SIZE + CELL_SIZE // 2
            pygame.draw.line(screen, (255, 40, 40), (sx, sy), (ex, ey), 2)


_SPRINGS_SURFACE: Optional[pygame.Surface] = None
_SPRINGS_DIRTY = True


def render_springs(screen: pygame.Surface, spring_positions: Set[Tuple[int, int]]) -> None:
    """Green diamond marker with green overlay showing SPRING_HEAL_RANGE radius (cached)."""
    global _SPRINGS_SURFACE, _SPRINGS_DIRTY
    if not spring_positions:
        return
    if _SPRINGS_SURFACE is not None and not _SPRINGS_DIRTY:
        screen.blit(_SPRINGS_SURFACE, (0, 0))
        return
    _SPRINGS_SURFACE = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    for sx, sy in spring_positions:
        px = sx * CELL_SIZE + CELL_SIZE // 2
        py = sy * CELL_SIZE + CELL_SIZE // 2
        # Green overlay circle showing heal range (SPRING_HEAL_RANGE cells)
        radius = SPRING_HEAL_RANGE * CELL_SIZE
        circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(circle_surf, (40, 200, 80, SPRING_ZONE_ALPHA), (radius, radius), radius)
        _SPRINGS_SURFACE.blit(circle_surf, (px - radius, py - radius))
        # Green diamond at center
        size = CELL_SIZE // 2
        points = [(px, py - size), (px + size, py), (px, py + size), (px - size, py)]
        pygame.draw.polygon(_SPRINGS_SURFACE, COLOR_SPRING, points)
        pygame.draw.polygon(_SPRINGS_SURFACE, (20, 120, 40), points, 2)
    _SPRINGS_DIRTY = False
    screen.blit(_SPRINGS_SURFACE, (0, 0))


def render_controls_hint(screen: pygame.Surface) -> None:
    font = pygame.font.Font(None, 20)
    hints = [
        "SPACE: Pause/Play",
        "RETURN: Skip",
        "+/-: Speed",
        "F: Toggle visited",
        "I: Toggle info",
        "0: Show/hide all",
        "1-5: Toggle agent",
        "R: Reset",
        "Q: Quit",
    ]
    block_h = 20 * len(hints) + 10
    x = WINDOW_SIZE - 155
    y = WINDOW_SIZE - block_h - 10

    bg = pygame.Surface((150, block_h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 150))
    screen.blit(bg, (x, y))

    for hint in hints:
        screen.blit(font.render(hint, True, (180, 180, 180)), (x + 8, y + 4))
        y += 20


# ── Terrain Helpers ─────────────────────────────────────────────────────────

def _try_generate_terrain(terrain_grid: TerrainGrid) -> None:
    try:
        terrain_grid.generate_random(obstacle_pct=0.25, spawn_margin=3)
    except ImportError:
        terrain_grid.clear()
        rng = random.Random()
        types = [TerrainType.OBSTACLE, TerrainType.FIRE, TerrainType.FOREST, TerrainType.WATER]
        weights = [0.25, 0.03, 0.10, 0.05]
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                r = rng.random()
                cumulative = 0.0
                for t, w in zip(types, weights):
                    cumulative += w
                    if r < cumulative:
                        terrain_grid.set(x, y, t)
                        break


def _add_fire_patches(terrain_grid: TerrainGrid, fire_pct: float = 0.06) -> None:
    """Add organic lava-lake fire patches using pixel-level Perlin noise.

    Works at 256x256 resolution (matching generate_random) and upscales 4x
    to 1024x1024 for smooth, natural shapes instead of blocky cell-level circles.
    """
    gen_size = terrain_grid.pixel_width // 4  # 256
    upscale = 4

    fire_noise = TerrainGrid._sample_noise(
        gen_size,
        seed=random.randint(0, 65535),
        scale=25.0,   # low scale → large blob features (lava lakes)
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
    )

    # Downsample existing grid to 256x256 (sample every 4th pixel)
    grid_256 = terrain_grid.grid[::upscale, ::upscale].copy()
    empty_mask = grid_256 == TerrainType.EMPTY

    if not empty_mask.any():
        return

    # Target fire_pct of total area, placed only on empty cells
    target_count = int(gen_size * gen_size * fire_pct)
    empty_vals = fire_noise[empty_mask]
    if target_count >= len(empty_vals):
        threshold = float(empty_vals.min()) - 1.0
    else:
        pct = max(0.0, 100.0 - target_count / len(empty_vals) * 100.0)
        threshold = float(np.percentile(empty_vals, pct))

    fire_mask_256 = empty_mask & (fire_noise > threshold)

    # Upscale fire mask to 1024x1024 and merge — only overwrite empty pixels
    fire_mask_full = np.kron(fire_mask_256, np.ones((upscale, upscale), dtype=bool))
    empty_full = terrain_grid.grid == TerrainType.EMPTY
    terrain_grid.grid[fire_mask_full & empty_full] = TerrainType.FIRE
    terrain_grid._surface_dirty = True


_WATER_DEPTH_SURFACE: Optional[pygame.Surface] = None
_WATER_DEPTH_DIRTY = True


def render_water_depth(screen: pygame.Surface, terrain_grid: TerrainGrid) -> None:
    """Overlay deep water pixels with darker blue based on distance from shore."""
    global _WATER_DEPTH_SURFACE, _WATER_DEPTH_DIRTY

    if _WATER_DEPTH_SURFACE is not None and not _WATER_DEPTH_DIRTY:
        screen.blit(_WATER_DEPTH_SURFACE, (0, 0))
        return

    pw = terrain_grid.pixel_width
    ph = terrain_grid.pixel_height

    # Pixel-level water mask straight from the terrain grid (1024x1024)
    water_full = terrain_grid.grid == TerrainType.WATER

    if not water_full.any():
        _WATER_DEPTH_SURFACE = pygame.Surface((pw, ph), pygame.SRCALPHA)
        _WATER_DEPTH_DIRTY = False
        return

    try:
        from scipy.ndimage import distance_transform_edt
        # Euclidean distance from each water pixel to nearest non-water pixel
        dist_full = distance_transform_edt(water_full).astype(np.float32)
    except ImportError:
        # Fallback: BFS at 256x256, upscale 4x to 1024x1024
        scale = 4
        sz = pw // scale  # 256
        water_small = water_full[::scale, ::scale]
        dist_small = np.full((sz, sz), -1, dtype=np.int32)
        q: deque = deque()
        for x in range(sz):
            for y in range(sz):
                if not water_small[x, y]:
                    continue
                is_shore = False
                for ddx, ddy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                    nx2, ny2 = x + ddx, y + ddy
                    if nx2 < 0 or nx2 >= sz or ny2 < 0 or ny2 >= sz:
                        is_shore = True
                        break
                    if not water_small[nx2, ny2]:
                        is_shore = True
                        break
                if is_shore:
                    dist_small[x, y] = 0
                    q.append((x, y))
        while q:
            cx, cy = q.popleft()
            for ddx, ddy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx2, ny2 = cx + ddx, cy + ddy
                if 0 <= nx2 < sz and 0 <= ny2 < sz and water_small[nx2, ny2] and dist_small[nx2, ny2] < 0:
                    dist_small[nx2, ny2] = dist_small[cx, cy] + 1
                    q.append((nx2, ny2))
        # Upscale 256→1024 via nearest-neighbor repeat
        dist_full = np.kron(dist_small.astype(np.float32), np.ones((scale, scale), dtype=np.float32))

    max_dist = float(dist_full.max())
    if max_dist < 1.0:
        max_dist = 1.0

    # Build overlay using surfarray for pixel-level smoothness
    _WATER_DEPTH_SURFACE = pygame.Surface((pw, ph), pygame.SRCALPHA)
    arr = pygame.surfarray.pixels3d(_WATER_DEPTH_SURFACE)
    alpha_arr = pygame.surfarray.pixels_alpha(_WATER_DEPTH_SURFACE)

    depth_mask = dist_full > 0
    alpha_values = (dist_full / max_dist * 100).clip(0, 255).astype(np.uint8)

    arr[depth_mask, 0] = 0
    arr[depth_mask, 1] = 0
    arr[depth_mask, 2] = 60
    alpha_arr[depth_mask] = alpha_values[depth_mask]

    del arr, alpha_arr  # release surfarray locks

    _WATER_DEPTH_DIRTY = False
    screen.blit(_WATER_DEPTH_SURFACE, (0, 0))


def generate_terrain(terrain_grid: TerrainGrid) -> None:
    global _WATER_DEPTH_DIRTY
    _WATER_DEPTH_DIRTY = True

    _try_generate_terrain(terrain_grid)

    # Extra fire patches so Dijkstra/A* have HP-costly cells to route around
    _add_fire_patches(terrain_grid)

    # Clear area around start
    sx, sy = START
    half = SPAWN_CLEAR_RADIUS // 2
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                terrain_grid.set(nx, ny, TerrainType.EMPTY)


def _has_los(
    terrain_grid: TerrainGrid, x0: int, y0: int, x1: int, y1: int,
) -> bool:
    """DDA line-of-sight: True if no obstacle blocks the line from (x0,y0) to (x1,y1)."""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    x, y = x0, y0

    if dx >= dy:
        err = dx // 2
        while x != x1:
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
            if (x, y) != (x1, y1) and not terrain_grid.is_walkable(x, y):
                return False
    else:
        err = dy // 2
        while y != y1:
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
            if (x, y) != (x1, y1) and not terrain_grid.is_walkable(x, y):
                return False
    return True


def _passage_width(
    terrain_grid: TerrainGrid, x: int, y: int,
) -> Tuple[int, int]:
    """Return (min_span, max_span) of the corridor at (x, y).

    Horizontal span = consecutive walkable cells left+right through (x,y).
    Vertical span = consecutive walkable cells up+down through (x,y).
    """
    # Horizontal span
    h_span = 1
    nx = x - 1
    while nx >= 0 and terrain_grid.is_walkable(nx, y):
        h_span += 1
        nx -= 1
    nx = x + 1
    while nx < GRID_SIZE and terrain_grid.is_walkable(nx, y):
        h_span += 1
        nx += 1

    # Vertical span
    v_span = 1
    ny = y - 1
    while ny >= 0 and terrain_grid.is_walkable(x, ny):
        v_span += 1
        ny -= 1
    ny = y + 1
    while ny < GRID_SIZE and terrain_grid.is_walkable(x, ny):
        v_span += 1
        ny += 1

    return min(h_span, v_span), max(h_span, v_span)


def _place_outposts(
    terrain_grid: TerrainGrid,
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Place NUM_OUTPOSTS at corridor chokepoints, return (outpost_positions, danger_zone).

    Outposts avoid the start-clear area and enforce MIN_OUTPOST_DISTANCE (Euclidean)
    between each other.  The danger zone is all walkable cells within
    OUTPOST_FIRE_RANGE Euclidean distance with clear line-of-sight.
    """
    outposts: Set[Tuple[int, int]] = set()
    sx, sy = START
    clear_r = SPAWN_CLEAR_RADIUS + OUTPOST_FIRE_RANGE  # keep danger zone away from start

    # Build scored candidates: walkable cells outside start-clear area
    scored: List[Tuple[float, int, int]] = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if not terrain_grid.is_walkable(x, y):
                continue
            # Skip cells near map edges
            if x < OUTPOST_EDGE_MARGIN or x >= GRID_SIZE - OUTPOST_EDGE_MARGIN:
                continue
            if y < OUTPOST_EDGE_MARGIN or y >= GRID_SIZE - OUTPOST_EDGE_MARGIN:
                continue
            dist_to_start = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if dist_to_start <= clear_r:
                continue
            # Skip cells on or adjacent to fire/lava
            near_fire = False
            for fdx in range(-2, 3):
                for fdy in range(-2, 3):
                    fx, fy = x + fdx, y + fdy
                    if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE:
                        if terrain_grid.get(fx, fy) == TerrainType.FIRE:
                            near_fire = True
                            break
                if near_fire:
                    break
            if near_fire:
                continue
            min_span, max_span = _passage_width(terrain_grid, x, y)
            # Lower score = narrower corridor = better chokepoint
            score = min_span * 100 - max_span + random.random()
            scored.append((score, x, y))

    scored.sort()

    for score, cx, cy in scored:
        if len(outposts) >= NUM_OUTPOSTS:
            break
        # Enforce minimum inter-outpost Euclidean distance
        if any(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < MIN_OUTPOST_DISTANCE
               for ox, oy in outposts):
            continue
        outposts.add((cx, cy))

    # Compute danger zone: walkable cells within OUTPOST_FIRE_RANGE with LOS
    r = OUTPOST_FIRE_RANGE
    r_sq = r * r
    danger: Set[Tuple[int, int]] = set()
    for cx, cy in outposts:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy > r_sq:
                    continue
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    continue
                if not terrain_grid.is_walkable(nx, ny):
                    continue
                if _has_los(terrain_grid, cx, cy, nx, ny):
                    danger.add((nx, ny))

    return outposts, danger


def _place_springs(
    terrain_grid: TerrainGrid,
    outpost_positions: Set[Tuple[int, int]],
) -> Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]:
    """Place NUM_SPRINGS healing springs, return (spring_positions, heal_zone).

    Springs must be on walkable cells with forest or water neighbors, no fire
    or obstacles within 2 cells, and away from outposts.  The heal_zone is all
    walkable cells within SPRING_HEAL_RANGE Euclidean distance of any spring.
    """
    springs: Set[Tuple[int, int]] = set()
    sx, sy = START
    clear_r = SPAWN_CLEAR_RADIUS + 2

    scored: List[Tuple[float, int, int]] = []
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if not terrain_grid.is_walkable(x, y):
                continue
            if abs(x - sx) + abs(y - sy) <= clear_r:
                continue
            # Must have at least one forest and one water neighbor (4-connected)
            has_forest = False
            has_water = False
            for dx, dy in [(0, -1), (0, 1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    t = terrain_grid.get(nx, ny)
                    if t == TerrainType.FOREST:
                        has_forest = True
                    elif t == TerrainType.WATER:
                        has_water = True
            if not (has_forest and has_water):
                continue
            # No fire or obstacles within 2 cells
            bad = False
            for fdx in range(-2, 3):
                for fdy in range(-2, 3):
                    fx, fy = x + fdx, y + fdy
                    if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE:
                        t = terrain_grid.get(fx, fy)
                        if t == TerrainType.FIRE or t == TerrainType.OBSTACLE:
                            bad = True
                            break
                if bad:
                    break
            if bad:
                continue
            # Prefer narrow corridors
            min_span, max_span = _passage_width(terrain_grid, x, y)
            score = min_span * 100 - max_span + random.random()
            scored.append((score, x, y))

    scored.sort()

    for score, cx, cy in scored:
        if len(springs) >= NUM_SPRINGS:
            break
        # Away from outposts
        if any(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 < 6
               for ox, oy in outpost_positions):
            continue
        # Enforce distance between springs
        if any(((cx - sx2) ** 2 + (cy - sy2) ** 2) ** 0.5 < MIN_SPRING_DISTANCE
               for sx2, sy2 in springs):
            continue
        springs.add((cx, cy))

    # Compute healing zone: walkable cells within SPRING_HEAL_RANGE of any spring
    r = SPRING_HEAL_RANGE
    r_sq = r * r
    heal_zone: Set[Tuple[int, int]] = set()
    for cx, cy in springs:
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy > r_sq:
                    continue
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and terrain_grid.is_walkable(nx, ny):
                    heal_zone.add((nx, ny))

    return springs, heal_zone


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    env = GridWorld(grid_size=GRID_SIZE, render_mode="human")
    terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
    generate_terrain(terrain_grid)

    state = AppState()

    # Place outposts and populate state (mutable sets — wrapper keeps a reference)
    outposts, danger = _place_outposts(terrain_grid)
    state.outpost_positions = outposts
    state.danger_zone = danger
    springs, heal = _place_springs(terrain_grid, outposts)
    state.spring_positions = springs
    state.heal_zone = heal

    # Reward wrapper (computes combat-readiness cost for informed search)
    # Shares the mutable danger_zone set so resets propagate automatically.
    reward_wrapper = CombatReadinessRewardWrapper(
        CardinalActionWrapper(FullyObservableWrapper(env, terrain_grid)),
        terrain_grid,
        danger_zone=state.danger_zone,
        heal_zone=state.heal_zone,
    )

    def make_agents(
        goal: Optional[Tuple[int, int]],
        prev_agents: Optional[List[SearchAgent]] = None,
    ) -> List[SearchAgent]:
        if goal is None:
            return []
        kw = dict(danger_zone=state.danger_zone, heal_zone=state.heal_zone)
        agent_classes = [DFSAgent, BFSAgent, DijkstraAgent, AStarAgent, BeamSearchAgent]
        new_list: List[SearchAgent] = []
        for i, cls in enumerate(agent_classes):
            if prev_agents and i < len(prev_agents):
                start_pos = prev_agents[i].position
                agent = cls(start_pos, goal, terrain_grid, **kw)
                agent.stats = prev_agents[i].stats  # carry stats forward
            else:
                agent = cls(START, goal, terrain_grid, **kw)
            new_list.append(agent)
        return new_list

    agents: List[SearchAgent] = []

    vis_env = SearchVisualiserWrapper(reward_wrapper, agents)
    vis_env.reset()
    env._init_pygame()
    pygame.display.set_caption("State Space Search -- DFS / BFS / Dijkstra / A* / Beam")

    screen = env.screen
    clock = env.clock

    while state.running:
        # ── Events ──
        new_agents = handle_events(state, agents, terrain_grid, vis_env, make_agents)
        if new_agents is not None:
            agents = new_agents

        # ── Skip (runs to completion) ──
        if state.skip_requested and agents:
            for agent in agents:
                while not agent.is_done:
                    actions = agent.get_actions()
                    results = []
                    for pos, act in actions:
                        reward_wrapper.set_position(pos)
                        _, reward, _, _, info = reward_wrapper.step(act)
                        results.append((reward, info))
                    agent.step(results)
            state.skip_requested = False
            state.paused = True
            # Build visited caches for all agents after skip
            for agent in agents:
                if agent.search_done:
                    vis_env._build_final_cache(agent)

        # ── Normal tick ──
        if not state.paused and agents:
            for _ in range(state.speed):
                for agent in agents:
                    was_searching = not agent.search_done
                    actions = agent.get_actions()
                    results = []
                    for pos, act in actions:
                        reward_wrapper.set_position(pos)
                        _, reward, _, _, info = reward_wrapper.step(act)
                        results.append((reward, info))
                    agent.step(results)
                    # Build cache the moment search completes
                    if was_searching and agent.search_done:
                        vis_env._build_final_cache(agent)

        # ── Render ──
        render_background(screen)
        render_terrain(screen, terrain_grid)
        render_water_depth(screen, terrain_grid)
        render_danger_zone(screen, state.danger_zone)
        render_grid_lines(screen)

        if agents:
            vis_env.render_search_overlays(screen, state.visible, state.show_visited)

        render_start_marker(screen)

        if state.goal is not None:
            render_goal(screen, state.goal)

        if agents:
            render_agent_markers(screen, agents, state.visible)

        render_outposts(screen, state.outpost_positions)
        render_springs(screen, state.spring_positions)
        if agents:
            render_outpost_fire(screen, state.outpost_positions, agents, state.visible, terrain_grid)

        if state.show_info:
            if agents:
                render_stats_overlay(screen, agents, state.visible, state.speed, state.paused)
                render_legend(screen, agents, state.visible)
                if all(a.is_done for a in agents):
                    render_next_goal_prompt(screen)
            elif state.waiting_for_goal:
                render_waiting_prompt(screen)

            render_controls_hint(screen)

        pygame.display.flip()
        clock.tick(FPS)

    env.close()


if __name__ == "__main__":
    main()
