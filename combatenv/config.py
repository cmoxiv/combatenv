"""
Configuration parameters for the Grid-World Multi-Agent Tactical Simulation.

This module centralizes all tunable parameters for the simulation, organized
into logical groups. Modifying these values allows customization of game
mechanics, visuals, and performance characteristics without code changes.

Parameter Categories:
    - Window and Grid: Display size, cell dimensions, grid resolution
    - Agent Configuration: Team size, visual proportions
    - Field of View: Legacy single-layer FOV settings
    - Movement: Speed, rotation, wandering behavior
    - Boundary and Collision: Margins, detection thresholds, spacing
    - Colors: RGB values for all visual elements
    - FOV Alpha Values: Transparency for FOV visualization layers
    - Frame Rate: Target FPS for simulation
    - Combat FOV Layers: Two-tier FOV system (near/far) with accuracy
    - Combat Projectiles: Speed, damage, lifetime, collision radius
    - Combat Agent: Health, cooldowns, friendly fire settings
    - Resource Management: Stamina, armor, and ammo systems

Units:
    - Distances are in grid cells (1 cell = CELL_SIZE pixels)
    - Speeds are in grid cells per second
    - Angles are in degrees
    - Times are in seconds
    - Alpha values are 0-255 (higher = more opaque)

Performance Notes:
    - GRID_SIZE is derived from WINDOW_SIZE / CELL_SIZE
    - Larger FOV ranges increase computation time
    - More agents increase collision detection overhead (mitigated by spatial grid)

Example:
    >>> from config import AGENT_MOVE_SPEED, GRID_SIZE
    >>> print(f"Agents move at {AGENT_MOVE_SPEED} cells/second in a {GRID_SIZE}x{GRID_SIZE} grid")
    Agents move at 3.0 cells/second in a 64x64 grid
"""

# Window and Grid Configuration
WINDOW_SIZE = 1024  # Window resolution (excluding title bar and borders)
CELL_SIZE = 16      # Size of each grid cell in pixels
GRID_SIZE = WINDOW_SIZE // CELL_SIZE  # Derived: 64x64 grid

# Agent Configuration
NUM_AGENTS_PER_TEAM = 32   # Number of agents per team
AGENT_SIZE_RATIO = 0.7     # Agent circle size as percentage of cell size
AGENT_NOSE_RATIO = 0.4     # Nose line length as percentage of cell size

# Field of View Configuration
LOS = 3              # Line of sight range in grid cells
FOV_ANGLE = 90       # Field of view angle in degrees

# Movement Configuration
AGENT_MOVE_SPEED = 3.0       # Grid cells per second (for smooth movement)
AGENT_ROTATION_SPEED = 180.0 # Degrees per second (for smooth rotation)
WANDER_DIRECTION_CHANGE = 0.02  # Probability of changing direction each frame

# Boundary and Collision Configuration
BOUNDARY_MARGIN = 0.5                  # Minimum distance from grid edge
BOUNDARY_DETECTION_THRESHOLD = 1.0     # Distance to trigger boundary avoidance
AGENT_SPAWN_SPACING = 1.0              # Minimum spacing between spawned agents
AGENT_COLLISION_RADIUS = 0.8           # Minimum separation distance between agents

# Colors (RGB)
COLOR_BACKGROUND = (235, 240, 200)  # Faint greenish yellow (empty terrain)
COLOR_GRID_LINES = (220, 220, 220)  # Lighter grey for tactical grid
COLOR_BLUE_TEAM = (0, 0, 255)       # Blue
COLOR_RED_TEAM = (255, 0, 0)        # Red
COLOR_PURPLE = (128, 0, 128)        # Purple (for FOV overlap)

# FOV Highlight Colors (RGBA with alpha for transparency)
FOV_BLUE_ALPHA = 30      # Alpha value for blue FOV highlighting (legacy)
FOV_RED_ALPHA = 30       # Alpha value for red FOV highlighting (legacy)
FOV_PURPLE_ALPHA = 40    # Alpha value for purple FOV overlap (legacy)

# Combat FOV Layer Alpha Values
FOV_NEAR_ALPHA = 50      # Alpha for near FOV (darker)
FOV_FAR_ALPHA = 25       # Alpha for far FOV (lighter)

# FOV Overlap Alpha Values (by layer combination)
OVERLAP_NEAR_NEAR_ALPHA = 80   # Both teams near FOV (darkest purple)
OVERLAP_MIXED_ALPHA = 60        # One near, one far (medium purple)
OVERLAP_FAR_FAR_ALPHA = 40      # Both teams far FOV (lightest purple)

# Frame Rate
FPS = 60

# Grid Line Width
GRID_LINE_WIDTH = 1  # Pixel width of grid lines

# Combat Configuration - FOV Layers
NEAR_FOV_RANGE = 3.0          # Grid cells for near FOV layer
NEAR_FOV_ANGLE = 90.0         # Degrees for near FOV cone
NEAR_FOV_ACCURACY = 0.99      # 99% accuracy in near FOV

FAR_FOV_RANGE = 5.0           # Grid cells for far FOV layer
FAR_FOV_ANGLE = 120.0         # Degrees for far FOV cone
FAR_FOV_ACCURACY = 0.80       # 80% accuracy in far FOV

# Combat Configuration - Movement Penalty
MOVEMENT_ACCURACY_PENALTY = 0.5  # Accuracy multiplier when moving (50% of base accuracy)

# Combat Configuration - Projectiles
PROJECTILE_SPEED = 15.0       # Grid cells per second
PROJECTILE_DAMAGE = 25        # HP damage per hit
PROJECTILE_RANGE = FAR_FOV_RANGE * 2  # Max distance = 2x far FOV range (10 cells)
PROJECTILE_LIFETIME = PROJECTILE_RANGE / PROJECTILE_SPEED  # Derived from range/speed
PROJECTILE_RADIUS = 0.3       # Collision detection radius

# Combat Configuration - Agent Combat
AGENT_MAX_HEALTH = 100        # Maximum health points
SHOOT_COOLDOWN = 0.5          # Seconds between shots
FRIENDLY_FIRE_ENABLED = True  # Allow teammates to damage each other

# Resource Management - Stamina
AGENT_MAX_STAMINA = 100.0          # Maximum stamina points
STAMINA_REGEN_RATE_IDLE = 20.0     # Stamina per second when not moving
STAMINA_REGEN_RATE_MOVING = 5.0    # Stamina per second while moving
STAMINA_DRAIN_RATE = 15.0          # Stamina consumed per second of movement
LOW_STAMINA_THRESHOLD = 20.0       # Threshold for movement penalty
MOVEMENT_SPEED_PENALTY_LOW_STAMINA = 0.5  # Speed multiplier at low stamina

# Resource Management - Armor
AGENT_MAX_ARMOR = 100          # Maximum armor points
ARMOR_REGEN_RATE = 0.0         # No regeneration (depleting resource)

# Resource Management - Ammo
AGENT_MAX_AMMO = 1000          # Total ammo reserve
MAGAZINE_SIZE = 30             # Rounds per magazine
RELOAD_TIME = 2.0              # Seconds to reload
AUTO_RELOAD_ON_EMPTY = True    # Auto-reload when magazine empty

# Combat Colors (RGB)
COLOR_PROJECTILE_BLUE = (50, 50, 255)    # Light blue for blue projectiles
COLOR_PROJECTILE_RED = (255, 50, 50)     # Light red for red projectiles
COLOR_DEAD_AGENT = (128, 128, 128)       # Gray for dead agents

# Combat Rendering
PROJECTILE_SIZE_RATIO = 0.15  # Projectile size relative to cell size (thin bullets)

# Muzzle Flash
MUZZLE_FLASH_OFFSET = 0.4     # Grid cells in front of agent to spawn flash
MUZZLE_FLASH_LIFETIME = 0.1   # Seconds the flash is visible

# Respawn
RESPAWN_DELAY_SECONDS = 1.0

# Terrain Configuration
FIRE_DAMAGE_PER_STEP = 2        # HP per step in fire (bypasses armor)
TERRAIN_OBSTACLE_PCT = 0.10     # 10% of grid for obstacles

# Operational Grid Configuration
OPERATIONAL_GRID_SIZE = 8       # 8x8 operational grid
TACTICAL_CELLS_PER_OPERATIONAL = 8  # 8x8 tactical cells per operational cell (64/8)
OPERATIONAL_GRID_COLOR = (150, 150, 150)  # Lighter black for operational grid
OPERATIONAL_GRID_LINE_WIDTH = 2  # Medium lines for operational grid
MAJOR_TERRAIN_PCT = 0.80        # 80% of cell is major terrain
MINOR_TERRAIN_PCT = 0.20        # 20% of cell is minor terrain
EMPTY_WEIGHT = 3.0              # Empty terrain 3x more likely than others

# Terrain Generation Grid (finer than operational grid for natural-looking terrain)
TERRAIN_GRID_SIZE = 16              # 16x16 terrain blocks
TACTICAL_CELLS_PER_TERRAIN_BLOCK = 4  # 4x4 tactical cells per terrain block

# Forest Configuration
FOREST_SPEED_MULTIPLIER = 0.5       # 50% speed in forest
FOREST_DETECTION_MULTIPLIER = 0.5   # 50% detection range for agents in forest

# Water Configuration
WATER_SPEED_MULTIPLIER = 0.5        # 50% speed in water
WATER_DETECTION_MULTIPLIER = 0.0    # 0% detection - full invisibility in water

# Terrain Colors (RGB)
COLOR_OBSTACLE = (64, 64, 64)   # Dark gray
COLOR_FIRE = (220, 120, 30)     # Orange (lava)
COLOR_FOREST = (0, 100, 50)     # Dark green
COLOR_WATER = (100, 170, 220)   # Light blue (slightly darker)

# =============================================================================
# Unit Configuration
# =============================================================================
NUM_UNITS_PER_TEAM = 8           # Number of units (squads) per team
AGENTS_PER_UNIT = 4              # Agents per unit (32 / 8 = 4)
UNIT_SPAWN_SPREAD = 3.0          # Max radius for initial unit spread (grid cells)
UNIT_COHESION_RADIUS = 4.0       # Radius to consider agent "grouped" with unit
UNIT_WAYPOINT_ARRIVAL_DIST = 2.0 # Distance to consider waypoint reached
UNIT_MIN_RADIUS = 2.0            # Minimum distance from unit centroid (avoid clumping)
UNIT_MAX_RADIUS = 6.0            # Maximum distance from unit centroid (hard leash)

# =============================================================================
# Boids Flocking Configuration
# =============================================================================
BOIDS_COHESION_WEIGHT = 1.0      # Steer toward unit centroid
BOIDS_SEPARATION_WEIGHT = 1.5    # Avoid crowding nearby squadmates
BOIDS_ALIGNMENT_WEIGHT = 0.5     # Match heading of nearby squadmates
BOIDS_WAYPOINT_WEIGHT = 2.0      # Steer toward unit waypoint
BOIDS_MAX_FORCE = 0.5            # Maximum steering force per frame
BOIDS_SEPARATION_RADIUS = 1.5    # Minimum distance between squadmates

# =============================================================================
# Formation Breaking Configuration
# =============================================================================
FORMATION_BREAK_COMBAT_RANGE = 5.0    # Break if enemy within this range
FORMATION_BREAK_HEALTH_THRESHOLD = 0.3  # Break if health below 30%

# =============================================================================
# Cohesion Reward Configuration (legacy - for reward wrappers)
# =============================================================================
COHESION_REWARD_BONUS = 0.5      # Survival bonus multiplier at centroid (1.5x)
COHESION_ISOLATION_PENALTY = 0.3 # Survival penalty when isolated (0.7x)

# =============================================================================
# Cohesion Combat Bonuses (when within UNIT_COHESION_RADIUS of centroid)
# =============================================================================
COHESION_SURVIVAL_BONUS = 0.10    # +10% survival reward
COHESION_ACCURACY_BONUS = 0.10    # +10% accuracy
COHESION_VISION_BONUS = 0.10      # +10% FOV range
COHESION_ARMOR_BONUS = 0.10       # +10% effective armor (damage reduction)

# =============================================================================
# Unit Stance Configuration
# =============================================================================
from enum import IntEnum

class UnitStance(IntEnum):
    """Stance modes affecting unit movement behavior."""
    AGGRESSIVE = 0   # Charge toward enemies
    DEFENSIVE = 1    # Stay near waypoint
    PATROL = 2       # Follow waypoint path

# =============================================================================
# Strategic Command Rewards
# =============================================================================
WAYPOINT_DISTANCE_REWARD_SCALE = 0.01   # Per-step reward based on proximity to waypoint
STANCE_COMPLIANCE_REWARD = 0.005        # Bonus for following stance behavior
MAX_WAYPOINT_REWARD_DISTANCE = 20.0     # Distance at which waypoint reward becomes 0

# =============================================================================
# Kill-Based Rewards
# =============================================================================
AGENT_KILL_REWARD = 10.0          # Reward per kill

# =============================================================================
# Strategic Grid Configuration
# =============================================================================
STRATEGIC_GRID_SIZE = 4          # 4x4 strategic grid
TACTICAL_CELLS_PER_STRATEGIC = 16  # 16x16 tactical cells per strategic cell
STRATEGIC_GRID_COLOR = (100, 100, 100)  # Light black for strategic grid
STRATEGIC_GRID_LINE_WIDTH = 3    # Thick lines for strategic grid

# =============================================================================
# Corridor Map Configuration
# =============================================================================
CORRIDOR_WIDTH = 2               # Width of corridors in cells
CORRIDOR_SPACING = 8             # Corridors every N cells (aligned with operational grid)

# =============================================================================
# Observation Space Configuration
# =============================================================================
NUM_NEARBY_ENEMIES = 5           # Number of nearest enemies in observation
NUM_NEARBY_ALLIES = 5            # Number of nearest allies in observation
MAX_FOV_CELLS = 38               # Maximum FOV cells in observation
NUM_TERRAIN_TYPES = 5            # EMPTY, OBSTACLE, FIRE, FOREST, WATER
OBS_SIZE = 89                    # Total observation size (was 88, +1 for Chebyshev distance)

# =============================================================================
# Action Space Configuration
# =============================================================================
ACTION_SIZE = 4                  # move_x, move_y, shoot, think

# =============================================================================
# Reward Configuration
# =============================================================================
FRIENDLY_FIRE_PENALTY = -5.0     # Penalty for hitting a teammate
