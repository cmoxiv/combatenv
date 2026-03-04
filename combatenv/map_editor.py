"""
Map Editor for Combat Environment.

A standalone pygame application for creating and editing terrain maps.
Maps are saved as compressed .npz files that can be loaded into the simulation.
Supports pixel-level painting with variable brush sizes and procedural terrain generation.

Controls:
    Mouse:
        Left click/drag: Paint selected terrain
        Right click/drag: Erase (paint Empty)
        Scroll wheel: Adjust brush size

    Keyboard:
        1-5: Select terrain type (1=Empty, 2=Obstacle, 3=Fire, 4=Forest, 5=Water)
        [ / ]: Decrease / increase brush size
        R: Generate random terrain
        S: Save map
        L: Load map
        C: Clear map
        G: Toggle grid lines
        Q (Shift+Q): Quit

Usage:
    combatenv-edit-terrain
"""

# Import tkinter BEFORE pygame to avoid macOS rendering issues
try:
    from tkinter import Tk, filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

import random

import numpy as np
import pygame
import subprocess
import platform


def _native_save_dialog(default_name: str = "map.npz") -> str:
    """Show native save dialog using platform-specific methods."""
    system = platform.system()

    if system == "Darwin":
        # macOS: use osascript
        script = f'''
        set theFile to choose file name with prompt "Save Map" default name "{default_name}"
        return POSIX path of theFile
        '''
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    elif system == "Linux":
        # Linux: try zenity, then kdialog
        for cmd in [
            ['zenity', '--file-selection', '--save', '--filename', default_name, '--file-filter', '*.npz'],
            ['kdialog', '--getsavefilename', '.', '*.npz']
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    return result.stdout.strip()
            except FileNotFoundError:
                continue
            except Exception:
                pass

    elif system == "Windows":
        # Windows: use PowerShell
        script = '''
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.SaveFileDialog
        $dialog.Filter = "Map files (*.npz)|*.npz|All files (*.*)|*.*"
        $dialog.FileName = "map.npz"
        if ($dialog.ShowDialog() -eq 'OK') { $dialog.FileName }
        '''
        try:
            result = subprocess.run(
                ['powershell', '-Command', script],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    return ""


def _native_open_dialog() -> str:
    """Show native open dialog using platform-specific methods."""
    system = platform.system()

    if system == "Darwin":
        # macOS: use osascript
        script = '''
        set theFile to choose file with prompt "Load Map"
        return POSIX path of theFile
        '''
        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

    elif system == "Linux":
        # Linux: try zenity, then kdialog
        for cmd in [
            ['zenity', '--file-selection', '--file-filter', '*.npz *.json'],
            ['kdialog', '--getopenfilename', '.', '*.npz *.json']
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    return result.stdout.strip()
            except FileNotFoundError:
                continue
            except Exception:
                pass

    elif system == "Windows":
        # Windows: use PowerShell
        script = '''
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Filter = "Map files (*.npz;*.json)|*.npz;*.json|All files (*.*)|*.*"
        if ($dialog.ShowDialog() -eq 'OK') { $dialog.FileName }
        '''
        try:
            result = subprocess.run(
                ['powershell', '-Command', script],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass

    return ""

from combatenv import TerrainGrid, TerrainType, save_map, load_map
from combatenv.config import (
    GRID_SIZE, CELL_SIZE, WINDOW_SIZE,
    COLOR_BACKGROUND as _CFG_COLOR_BACKGROUND,
    COLOR_OBSTACLE, COLOR_FIRE, COLOR_FOREST, COLOR_WATER,
)
from combatenv.renderer import (
    render_water_depth, render_forest_depth,
    render_lava_variation, render_mountain_elevation,
)
import combatenv.renderer as _renderer


# Editor configuration
PALETTE_WIDTH = 100
EDITOR_WIDTH = PALETTE_WIDTH + WINDOW_SIZE
EDITOR_HEIGHT = WINDOW_SIZE

BRUSH_SIZES = [2, 4, 8, 16, 32]

# Color lookup table for pixel-level surfarray rendering (indexed by TerrainType)
COLOR_LUT = np.array([
    _CFG_COLOR_BACKGROUND,  # EMPTY
    COLOR_OBSTACLE,
    COLOR_FIRE,
    COLOR_FOREST,
    COLOR_WATER,
], dtype=np.uint8)

# UI Colors
COLOR_GRID = (200, 200, 200)
COLOR_PALETTE_BG = (50, 50, 50)
COLOR_SELECTED = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BUTTON = (80, 80, 80)
COLOR_BUTTON_HOVER = (100, 100, 100)

# Terrain palette (button colors)
TERRAIN_COLORS = {
    TerrainType.EMPTY: _CFG_COLOR_BACKGROUND,
    TerrainType.OBSTACLE: COLOR_OBSTACLE,
    TerrainType.FIRE: COLOR_FIRE,
    TerrainType.FOREST: COLOR_FOREST,
    TerrainType.WATER: COLOR_WATER,
}

TERRAIN_NAMES = {
    TerrainType.EMPTY: "Empty",
    TerrainType.OBSTACLE: "Obstacle",
    TerrainType.FIRE: "Fire",
    TerrainType.FOREST: "Forest",
    TerrainType.WATER: "Water",
}


class MapEditor:
    """Standalone map editor application."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Combat Environment - Map Editor")

        self.screen = pygame.display.set_mode((EDITOR_WIDTH, EDITOR_HEIGHT))
        self.clock = pygame.time.Clock()

        # Offscreen surface for pixel-level terrain rendering
        self.terrain_surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))

        self.terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        self.selected_terrain = TerrainType.OBSTACLE
        self.show_grid = True
        self.running = True

        # Brush state
        self.brush_size_index = 2  # Start at 8px radius
        self.brush_radius = BRUSH_SIZES[self.brush_size_index]
        self._scroll_accum = 0.0
        self._reseed_brush_noise()

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Button rects for palette
        self.terrain_buttons = {}
        self.action_buttons = {}
        self._setup_buttons()

        # Hide tkinter root window (if available)
        if HAS_TKINTER:
            self.tk_root = Tk()
            self.tk_root.withdraw()

    def _setup_buttons(self):
        """Setup button rectangles for the palette."""
        y = 10
        button_height = 40
        button_margin = 5
        button_width = PALETTE_WIDTH - 20

        # Terrain buttons
        for terrain_type in TerrainType:
            self.terrain_buttons[terrain_type] = pygame.Rect(
                10, y, button_width, button_height
            )
            y += button_height + button_margin

        y += 20  # Extra spacing before action buttons

        # Action buttons
        for action in ["Generate", "Save", "Load", "Clear"]:
            self.action_buttons[action] = pygame.Rect(
                10, y, button_width, button_height
            )
            y += button_height + button_margin

    def run(self):
        """Main editor loop."""
        while self.running:
            self._handle_events()
            self._render()
            self.clock.tick(60)

        pygame.quit()

    def _handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button in (1, 3):
                    self._reseed_brush_noise()
                self._handle_mouse_click(event)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    self._invalidate_overlays()

            elif event.type == pygame.MOUSEWHEEL:
                self._scroll_accum += event.y
                if self._scroll_accum >= 1.0:
                    self._change_brush_size(1)
                    self._scroll_accum = 0.0
                elif self._scroll_accum <= -1.0:
                    self._change_brush_size(-1)
                    self._scroll_accum = 0.0

        # Handle continuous painting while mouse is held
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] or mouse_buttons[2]:
            mouse_pos = pygame.mouse.get_pos()
            self._paint_at_position(mouse_pos, erase=mouse_buttons[2])

    def _handle_keydown(self, event):
        """Handle keyboard input."""
        if event.key == pygame.K_q and (event.mod & pygame.KMOD_SHIFT):
            self.running = False

        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid

        elif event.key == pygame.K_r:
            self._generate_terrain()

        elif event.key == pygame.K_s:
            self._save_map()

        elif event.key == pygame.K_l:
            self._load_map()

        elif event.key == pygame.K_c:
            self.terrain_grid.clear()
            self._invalidate_overlays()

        elif event.key == pygame.K_LEFTBRACKET:
            self._change_brush_size(-1)
        elif event.key == pygame.K_RIGHTBRACKET:
            self._change_brush_size(1)

        # Number keys for terrain selection
        elif event.key == pygame.K_1:
            self.selected_terrain = TerrainType.EMPTY
        elif event.key == pygame.K_2:
            self.selected_terrain = TerrainType.OBSTACLE
        elif event.key == pygame.K_3:
            self.selected_terrain = TerrainType.FIRE
        elif event.key == pygame.K_4:
            self.selected_terrain = TerrainType.FOREST
        elif event.key == pygame.K_5:
            self.selected_terrain = TerrainType.WATER

    def _handle_mouse_click(self, event):
        """Handle mouse click events."""
        mouse_pos = event.pos

        # Check terrain buttons
        for terrain_type, rect in self.terrain_buttons.items():
            if rect.collidepoint(mouse_pos):
                self.selected_terrain = terrain_type
                return

        # Check action buttons
        for action, rect in self.action_buttons.items():
            if rect.collidepoint(mouse_pos):
                if action == "Generate":
                    self._generate_terrain()
                elif action == "Save":
                    self._save_map()
                elif action == "Load":
                    self._load_map()
                elif action == "Clear":
                    self.terrain_grid.clear()
                    self._invalidate_overlays()
                return

        # Paint on grid
        if event.button == 1:  # Left click
            self._paint_at_position(mouse_pos, erase=False)
        elif event.button == 3:  # Right click
            self._paint_at_position(mouse_pos, erase=True)

    def _invalidate_overlays(self):
        """Mark all renderer overlay caches as dirty so they rebuild."""
        _renderer._WATER_DEPTH_DIRTY = True
        _renderer._FOREST_DEPTH_DIRTY = True
        _renderer._LAVA_DIRTY = True
        _renderer._MOUNTAIN_DIRTY = True

    def _reseed_brush_noise(self):
        """Generate a Perlin noise field used for organic brush edges."""
        # Sample at 256x256, upscale 4x to 1024x1024
        small = WINDOW_SIZE // 4
        try:
            field = TerrainGrid._sample_noise(
                small, seed=random.randint(0, 65535),
                scale=12.0, octaves=3, persistence=0.5, lacunarity=2.0,
            )
        except ImportError:
            field = np.random.default_rng().uniform(
                -0.5, 0.5, (small, small)
            ).astype(np.float32)
        full = np.kron(field, np.ones((4, 4), dtype=np.float32))
        # Normalize to 0..1
        nmin, nmax = float(full.min()), float(full.max())
        if nmax - nmin < 1e-6:
            nmax = nmin + 1.0
        self._brush_noise = (full - nmin) / (nmax - nmin)

    def _change_brush_size(self, direction):
        """Change brush size by stepping through BRUSH_SIZES."""
        self.brush_size_index = max(0, min(len(BRUSH_SIZES) - 1,
                                           self.brush_size_index + direction))
        self.brush_radius = BRUSH_SIZES[self.brush_size_index]

    def _generate_terrain(self):
        """Generate random terrain using Perlin noise."""
        self.terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        self.terrain_grid.generate_random(rng=random.Random())
        self._invalidate_overlays()

    def _paint_at_position(self, mouse_pos, erase=False):
        """Paint terrain using a Perlin noise-masked brush for organic edges."""
        mx, my = mouse_pos

        if mx < PALETTE_WIDTH:
            return

        px = mx - PALETTE_WIDTH
        py = my
        terrain = TerrainType.EMPTY if erase else self.selected_terrain
        r = self.brush_radius
        grid = self.terrain_grid
        noise = self._brush_noise

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                tx, ty = px + dx, py + dy
                if not (0 <= tx < WINDOW_SIZE and 0 <= ty < WINDOW_SIZE):
                    continue
                # Chebyshev distance, normalized 0..1
                dist = max(abs(dx), abs(dy)) / r
                if dist > 1.0:
                    continue
                # Threshold rises toward edge: center always paints, edge is noisy
                threshold = dist * dist
                if noise[tx, ty] > threshold:
                    grid.set_pixel(tx, ty, terrain)

    def _save_map(self):
        """Save the current map to a file."""
        if HAS_TKINTER:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".npz",
                filetypes=[("Map files", "*.npz"), ("All files", "*.*")],
                title="Save Map"
            )
        else:
            # Use native OS dialog (macOS/Linux/Windows)
            filepath = _native_save_dialog()

        # Fallback to default if dialog was cancelled or unavailable
        if not filepath:
            filepath = "map.npz"
            print(f"Using default filename: {filepath}")

        # Ensure .npz extension
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        save_map(self.terrain_grid, filepath)
        print(f"Map saved to: {filepath}")

    def _load_map(self):
        """Load a map from a file."""
        if HAS_TKINTER:
            filepath = filedialog.askopenfilename(
                filetypes=[("Map files", "*.npz *.json"), ("All files", "*.*")],
                title="Load Map"
            )
        else:
            # Use native OS dialog (macOS/Linux/Windows)
            filepath = _native_open_dialog()

        if not filepath:
            print("No file selected")
            return

        try:
            self.terrain_grid = load_map(filepath)
            self._invalidate_overlays()
            print(f"Map loaded from: {filepath}")
        except Exception as e:
            print(f"Error loading map: {e}")

    def _render(self):
        """Render the editor."""
        # Render palette
        self._render_palette()

        # Render terrain via surfarray
        self._render_terrain()

        if self.show_grid:
            self._render_grid_lines()

        # Render brush cursor
        self._render_brush_cursor()

        # Render instructions
        self._render_instructions()

        pygame.display.flip()

    def _render_brush_cursor(self):
        """Draw a circle at the mouse position showing the brush size."""
        mx, my = pygame.mouse.get_pos()
        if mx > PALETTE_WIDTH:
            pygame.draw.circle(self.screen, (255, 255, 255), (mx, my),
                               self.brush_radius, 1)

    def _render_palette(self):
        """Render the terrain palette."""
        # Palette background
        pygame.draw.rect(
            self.screen,
            COLOR_PALETTE_BG,
            (0, 0, PALETTE_WIDTH, EDITOR_HEIGHT)
        )

        # Terrain buttons
        for terrain_type, rect in self.terrain_buttons.items():
            color = TERRAIN_COLORS[terrain_type]

            # Draw button background
            pygame.draw.rect(self.screen, color, rect)

            # Draw selection highlight
            if terrain_type == self.selected_terrain:
                pygame.draw.rect(self.screen, COLOR_SELECTED, rect, 3)
            else:
                pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)

            # Draw label
            name = TERRAIN_NAMES[terrain_type]
            text_color = (0, 0, 0) if terrain_type == TerrainType.EMPTY else (255, 255, 255)
            text = self.small_font.render(name, True, text_color)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

        # Action buttons
        mouse_pos = pygame.mouse.get_pos()
        for action, rect in self.action_buttons.items():
            is_hover = rect.collidepoint(mouse_pos)
            color = COLOR_BUTTON_HOVER if is_hover else COLOR_BUTTON

            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (100, 100, 100), rect, 1)

            text = self.font.render(action, True, COLOR_TEXT)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)

    def _render_terrain(self):
        """Render the terrain grid at pixel level with color variation overlays."""
        rgb = COLOR_LUT[self.terrain_grid.grid]
        pygame.surfarray.blit_array(self.terrain_surface, rgb)
        render_water_depth(self.terrain_surface, self.terrain_grid)
        render_forest_depth(self.terrain_surface, self.terrain_grid)
        render_lava_variation(self.terrain_surface, self.terrain_grid)
        render_mountain_elevation(self.terrain_surface, self.terrain_grid)
        self.screen.blit(self.terrain_surface, (PALETTE_WIDTH, 0))

    def _render_grid_lines(self):
        """Render grid lines."""
        for i in range(GRID_SIZE + 1):
            # Vertical lines
            x = PALETTE_WIDTH + i * CELL_SIZE
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (x, 0),
                (x, EDITOR_HEIGHT)
            )

            # Horizontal lines
            y = i * CELL_SIZE
            pygame.draw.line(
                self.screen,
                COLOR_GRID,
                (PALETTE_WIDTH, y),
                (EDITOR_WIDTH, y)
            )

    def _render_instructions(self):
        """Render keyboard shortcuts and brush info at bottom of palette."""
        instructions = [
            f"Brush: {self.brush_radius}px",
            "",
            "R: Generate",
            "[/]: Brush size",
            "1-5: Terrain",
            "S/L: Save/Load",
            "C: Clear",
            "G: Grid",
            "Q: Quit",
        ]

        y = EDITOR_HEIGHT - len(instructions) * 18 - 10
        for line in instructions:
            color = (255, 255, 0) if line.startswith("Brush:") else (180, 180, 180)
            text = self.small_font.render(line, True, color)
            self.screen.blit(text, (10, y))
            y += 18


def main():
    """Entry point for the map editor."""
    editor = MapEditor()
    editor.run()


if __name__ == "__main__":
    main()
