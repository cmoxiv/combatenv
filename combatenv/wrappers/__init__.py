"""
Gymnasium wrappers for combatenv.

This module provides a layered wrapper architecture for different abstraction levels:

LAYERED ARCHITECTURE
====================

    +-----------------------------------------------------------+
    |                      USER / RL AGENT                      |
    +-----------------------------------------------------------+
                                |
    +-----------------------------------------------------------+
    | UTILITY WRAPPERS (stackable)                              |
    |   KeybindingsWrapper    - Keyboard/mouse input            |
    |   DebugOverlayWrapper   - Debug info rendering            |
    |   TerminalLogWrapper    - Console logging                 |
    |   TerrainGenWrapper     - Terrain generation              |
    +-----------------------------------------------------------+
                                |
    +-----------------------------------------------------------+
    | STRATEGIC LEVEL (grid control)                            |
    |   StrategicWrapper              - 4x4 grid observations   |
    |   StrategicDiscreteObsWrapper   - 2,916 strategic states  |
    |   StrategicDiscreteActionWrapper - 25 strategic actions   |
    +-----------------------------------------------------------+
                                |
    +-----------------------------------------------------------+
    | OPERATIONAL LEVEL (unit control)                          |
    |   OperationalWrapper              - Units, boids, waypts  |
    |   OperationalDiscreteObsWrapper   - 432 states per unit   |
    |   OperationalDiscreteActionWrapper - 19 unit actions      |
    +-----------------------------------------------------------+
                                |
    +-----------------------------------------------------------+
    | TACTICAL LEVEL (agent control)                            |
    |   TacticalWrapper               - Combat, terrain effects |
    |   DiscreteObservationWrapper    - 8,192 agent states      |
    |   DiscreteActionWrapper         - 10 agent actions        |
    +-----------------------------------------------------------+
                                |
    +-----------------------------------------------------------+
    |                  BASE ENVIRONMENT                         |
    |   TacticalCombatEnv - Core physics, agents, projectiles   |
    +-----------------------------------------------------------+

EXAMPLE USAGE
=============

Tactical RL Training (agent-level):
    env = TacticalCombatEnv(render_mode=None)
    env = MultiAgentWrapper(env)
    env = DiscreteObservationWrapper(env)
    env = DiscreteActionWrapper(env)

Operational RL Training (unit-level):
    env = TacticalCombatEnv(render_mode=None)
    env = OperationalWrapper(env)
    env = OperationalDiscreteObsWrapper(env, team="blue")
    env = OperationalDiscreteActionWrapper(env, team="blue")

Strategic RL Training (grid-level):
    env = TacticalCombatEnv(render_mode=None)
    env = OperationalWrapper(env)
    env = StrategicWrapper(env)
    env = StrategicDiscreteObsWrapper(env)
    env = StrategicDiscreteActionWrapper(env, team="blue")

Full Stack (visual demo):
    env = TacticalCombatEnv(render_mode="human")
    env = OperationalWrapper(env)
    env = StrategicWrapper(env)
    env = KeybindingsWrapper(env)
    env = DebugOverlayWrapper(env)

See example_layered_architecture.py for complete examples.
"""

# Base wrapper with attribute forwarding
from combatenv.wrappers.base_wrapper import BaseWrapper

# Base wrappers (GridWorld architecture)
from combatenv.wrappers.agent_wrapper import AgentWrapper
from combatenv.wrappers.team_wrapper import TeamWrapper
from combatenv.wrappers.terrain_wrapper import TerrainWrapper as BaseTerrainWrapper
from combatenv.wrappers.projectile_wrapper import ProjectileWrapper
from combatenv.wrappers.fov_wrapper import FOVWrapper as BaseFOVWrapper
from combatenv.wrappers.combat_wrapper import CombatWrapper as BaseCombatWrapper
from combatenv.wrappers.movement_wrapper import MovementWrapper
from combatenv.wrappers.observation_wrapper import ObservationWrapper as BaseObservationWrapper
from combatenv.wrappers.termination_wrapper import TerminationWrapper

# Multi-agent control
from combatenv.wrappers.multi_agent import MultiAgentWrapper

# Reward wrappers
from combatenv.wrappers.reward import RewardWrapper
from combatenv.wrappers.cohesion_reward import CohesionRewardWrapper
from combatenv.wrappers.chebyshev_reward import ChebyshevRewardWrapper
from combatenv.wrappers.unit_chebyshev_reward import UnitChebyshevWrapper
from combatenv.wrappers.goldilocks_reward import GoldilocksRewardWrapper
from combatenv.wrappers.anti_clump_reward import AntiClumpRewardWrapper
from combatenv.wrappers.fov_coverage_reward import FOVCoverageRewardWrapper

# Observation/action wrappers
from combatenv.wrappers.discrete_obs import DiscreteObservationWrapper
from combatenv.wrappers.discrete_action import DiscreteActionWrapper
from combatenv.wrappers.single_discrete_obs import SingleAgentDiscreteObsWrapper
from combatenv.wrappers.single_discrete_action import SingleAgentDiscreteActionWrapper
from combatenv.wrappers.action_mask import ActionMaskWrapper
from combatenv.wrappers.movement_obs import MovementObservationWrapper

# Unit system wrappers
from combatenv.wrappers.unit_wrapper import UnitWrapper

# Operational level (Phase 2)
from combatenv.wrappers.operational import OperationalWrapper

# Tactical level (Phase 3)
from combatenv.wrappers.tactical import TacticalWrapper

# Strategic level (Phase 4)
from combatenv.wrappers.strategic import StrategicWrapper

# Discrete wrappers by level (Phase 5)
# Operational level discrete wrappers
from combatenv.wrappers.operational_discrete_obs import OperationalDiscreteObsWrapper
from combatenv.wrappers.operational_discrete_action import OperationalDiscreteActionWrapper
# Operational training wrapper
from combatenv.wrappers.operational_training import OperationalTrainingWrapper
# Strategic level discrete wrappers
from combatenv.wrappers.strategic_discrete_obs import StrategicDiscreteObsWrapper
from combatenv.wrappers.strategic_discrete_action import StrategicDiscreteActionWrapper

# Utility wrappers (Phase 1)
from combatenv.wrappers.keybindings import KeybindingsWrapper
from combatenv.wrappers.debug_overlay import DebugOverlayWrapper
from combatenv.wrappers.terminal_log import TerminalLogWrapper
from combatenv.wrappers.terrain_gen import TerrainGenWrapper

# Environment modification
from combatenv.wrappers.empty_terrain import EmptyTerrainWrapper

# Action suppression (tactical level - replaces discrete actions)
from combatenv.wrappers.action_suppression import ActionSuppressionWrapper

# Rendering wrapper
from combatenv.wrappers.render_wrapper import RenderWrapper

# Task wrappers
from combatenv.wrappers.waypoint_task_wrapper import WaypointTaskWrapper

# Q-learning wrapper
from combatenv.wrappers.qlearning_wrapper import QLearningWrapper

# Tactical curriculum learning wrappers
from combatenv.wrappers.tactical_reward import TacticalRewardWrapper
from combatenv.wrappers.tactical_combat_reward import TacticalCombatRewardWrapper
from combatenv.wrappers.tactical_qlearning import TacticalQLearningWrapper

__all__ = [
    # Base wrapper
    "BaseWrapper",
    # Base wrappers (GridWorld architecture)
    "AgentWrapper",
    "TeamWrapper",
    "BaseTerrainWrapper",
    "ProjectileWrapper",
    "BaseFOVWrapper",
    "BaseCombatWrapper",
    "MovementWrapper",
    "BaseObservationWrapper",
    "TerminationWrapper",
    # Multi-agent
    "MultiAgentWrapper",
    # Reward wrappers
    "RewardWrapper",
    "CohesionRewardWrapper",
    "ChebyshevRewardWrapper",
    "UnitChebyshevWrapper",
    "GoldilocksRewardWrapper",
    "AntiClumpRewardWrapper",
    "FOVCoverageRewardWrapper",
    # Observation/action wrappers
    "DiscreteObservationWrapper",
    "DiscreteActionWrapper",
    "SingleAgentDiscreteObsWrapper",
    "SingleAgentDiscreteActionWrapper",
    "ActionMaskWrapper",
    "MovementObservationWrapper",
    # Unit system
    "UnitWrapper",
    # Operational level (Phase 2)
    "OperationalWrapper",
    # Tactical level (Phase 3)
    "TacticalWrapper",
    # Strategic level (Phase 4)
    "StrategicWrapper",
    # Discrete wrappers by level (Phase 5)
    "OperationalDiscreteObsWrapper",
    "OperationalDiscreteActionWrapper",
    "OperationalTrainingWrapper",
    "StrategicDiscreteObsWrapper",
    "StrategicDiscreteActionWrapper",
    # Utility wrappers (Phase 1)
    "KeybindingsWrapper",
    "DebugOverlayWrapper",
    "TerminalLogWrapper",
    "TerrainGenWrapper",
    # Environment modification
    "EmptyTerrainWrapper",
    # Action suppression
    "ActionSuppressionWrapper",
    # Rendering
    "RenderWrapper",
    # Task wrappers
    "WaypointTaskWrapper",
    # Q-learning
    "QLearningWrapper",
    # Tactical curriculum learning
    "TacticalRewardWrapper",
    "TacticalCombatRewardWrapper",
    "TacticalQLearningWrapper",
]
