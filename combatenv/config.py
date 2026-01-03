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
NUM_AGENTS_PER_TEAM = 100  # Number of agents per team
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
COLOR_BACKGROUND = (255, 255, 255)  # White
COLOR_GRID_LINES = (200, 200, 200)  # Light gray
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
PROJECTILE_SIZE_RATIO = 0.3   # Projectile size relative to cell size

# Muzzle Flash
MUZZLE_FLASH_OFFSET = 0.4     # Grid cells in front of agent to spawn flash
MUZZLE_FLASH_LIFETIME = 0.1   # Seconds the flash is visible

# Respawn
RESPAWN_DELAY_SECONDS = 1.0

# Terrain Configuration
FIRE_DAMAGE_PER_STEP = 2        # HP per step in fire (bypasses armor)
SWAMP_STUCK_MIN_STEPS = 30      # Min steps stuck (0.5 sec at 60 FPS)
SWAMP_STUCK_MAX_STEPS = 90      # Max steps stuck (1.5 sec at 60 FPS)
TERRAIN_BUILDING_PCT = 0.05     # 5% of grid
TERRAIN_FIRE_PCT = 0.02         # 2% of grid
TERRAIN_SWAMP_PCT = 0.03        # 3% of grid
TERRAIN_WATER_PCT = 0.03        # 3% of grid

# Terrain Colors (RGB)
COLOR_BUILDING = (64, 64, 64)   # Dark gray
COLOR_FIRE = (255, 100, 0)      # Orange
COLOR_SWAMP = (0, 100, 50)      # Dark green
COLOR_WATER = (0, 100, 200)     # Blue
