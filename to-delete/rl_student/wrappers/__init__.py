"""
Gymnasium wrappers for multi-agent Q-learning on combatenv.

NOTE: Wrappers have been moved to combatenv.wrappers for better organization.
This module re-exports them for backward compatibility.

Standard Wrapper Stack:
    TacticalCombatEnv (base - 1 controlled agent)
        |
    MultiAgentWrapper (enables 200-agent control)
        |
    RewardWrapper (optional - incentivizes movement to enemy corner)
        |
    DiscreteObservationWrapper (89 floats -> 2,048 discrete states)
        |
    DiscreteActionWrapper (4 floats -> 10 discrete actions)

Unit-Aware Wrapper Stack:
    TacticalCombatEnv (base)
        |
    MultiAgentWrapper (200-agent control)
        |
    UnitWrapper (unit tracking, boids, extended obs)
        |
    CohesionRewardWrapper (unit cohesion rewards)
        |
    RewardWrapper (optional)
        |
    DiscreteObservationWrapper
        |
    DiscreteActionWrapper
"""

# Re-export from combatenv.wrappers for backward compatibility
from combatenv.wrappers import (
    MultiAgentWrapper,
    RewardWrapper,
    DiscreteObservationWrapper,
    DiscreteActionWrapper,
    UnitWrapper,
    CohesionRewardWrapper,
    ChebyshevRewardWrapper,
    ActionMaskWrapper,
)

__all__ = [
    "MultiAgentWrapper",
    "RewardWrapper",
    "DiscreteObservationWrapper",
    "DiscreteActionWrapper",
    "UnitWrapper",
    "CohesionRewardWrapper",
    "ChebyshevRewardWrapper",
    "ActionMaskWrapper",
]
