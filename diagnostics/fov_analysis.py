"""
FOV Analysis Diagnostic Script

Calculates the maximum number of grid cells visible within near and far FOV cones.
"""

from combatenv.fov import get_fov_cells
from combatenv.config import NEAR_FOV_RANGE, NEAR_FOV_ANGLE, FAR_FOV_RANGE, FAR_FOV_ANGLE


def analyze_fov():
    """Analyze FOV cell counts for different orientations."""
    # Agent at center of grid
    pos = (32.0, 32.0)

    # Test multiple orientations to find max cells
    orientations = [0, 45, 90, 135, 180, 225, 270, 315]

    max_near = 0
    max_far = 0

    for orientation in orientations:
        near_cells = get_fov_cells(pos, orientation, NEAR_FOV_ANGLE, NEAR_FOV_RANGE)
        far_cells = get_fov_cells(pos, orientation, FAR_FOV_ANGLE, FAR_FOV_RANGE)
        max_near = max(max_near, len(near_cells))
        max_far = max(max_far, len(far_cells))

    # Use orientation 0 for detailed analysis
    orientation = 0.0
    near_cells = get_fov_cells(pos, orientation, NEAR_FOV_ANGLE, NEAR_FOV_RANGE)
    far_cells = get_fov_cells(pos, orientation, FAR_FOV_ANGLE, FAR_FOV_RANGE)

    print("=" * 50)
    print("FOV ANALYSIS")
    print("=" * 50)
    print()
    print(f"Near FOV Configuration:")
    print(f"  Range: {NEAR_FOV_RANGE} cells")
    print(f"  Angle: {NEAR_FOV_ANGLE} degrees")
    print(f"  Max cells: {max_near}")
    print()
    print(f"Far FOV Configuration:")
    print(f"  Range: {FAR_FOV_RANGE} cells")
    print(f"  Angle: {FAR_FOV_ANGLE} degrees")
    print(f"  Max cells: {max_far}")
    print()
    print(f"Combined Analysis (orientation={orientation}):")
    print(f"  Near FOV cells: {len(near_cells)}")
    print(f"  Far FOV cells: {len(far_cells)}")
    print(f"  Far-only cells: {len(far_cells - near_cells)}")
    print(f"  Total unique cells: {len(far_cells | near_cells)}")
    print()

    # Theoretical area calculation
    import math
    near_area = (NEAR_FOV_ANGLE / 360) * math.pi * (NEAR_FOV_RANGE ** 2)
    far_area = (FAR_FOV_ANGLE / 360) * math.pi * (FAR_FOV_RANGE ** 2)
    print(f"Theoretical sector areas:")
    print(f"  Near FOV: {near_area:.2f} square units")
    print(f"  Far FOV: {far_area:.2f} square units")
    print()


if __name__ == "__main__":
    analyze_fov()
