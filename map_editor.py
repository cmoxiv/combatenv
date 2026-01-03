"""
Map Editor for Combat Environment.

A standalone pygame application for creating and editing terrain maps.
Maps are saved as JSON files that can be loaded into the simulation.

Controls:
    Mouse:
        Left click/drag: Paint selected terrain
        Right click/drag: Erase (paint Empty)

    Keyboard:
        1-5: Select terrain type (1=Empty, 2=Building, 3=Fire, 4=Swamp, 5=Water)
        S: Save map
        L: Load map
        C: Clear map
        G: Toggle grid lines
        Q (Shift+Q): Quit

Usage:
    python map_editor.py
"""

# Import tkinter BEFORE pygame to avoid macOS rendering issues
try:
    from tkinter import Tk, filedialog
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

import pygame
import sys
import subprocess
import platform


def _native_save_dialog(default_name: str = "map.json") -> str:
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
            ['zenity', '--file-selection', '--save', '--filename', default_name, '--file-filter', '*.json'],
            ['kdialog', '--getsavefilename', '.', '*.json']
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
        $dialog.Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*"
        $dialog.FileName = "map.json"
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
        set theFile to choose file with prompt "Load Map" of type {"json", "public.json"}
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
            ['zenity', '--file-selection', '--file-filter', '*.json'],
            ['kdialog', '--getopenfilename', '.', '*.json']
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
        $dialog.Filter = "JSON files (*.json)|*.json|All files (*.*)|*.*"
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
    COLOR_BUILDING, COLOR_FIRE, COLOR_SWAMP, COLOR_WATER
)


# Editor configuration
PALETTE_WIDTH = 100
EDITOR_WIDTH = PALETTE_WIDTH + WINDOW_SIZE
EDITOR_HEIGHT = WINDOW_SIZE

# Colors
COLOR_BACKGROUND = (240, 240, 240)
COLOR_GRID = (200, 200, 200)
COLOR_EMPTY = (255, 255, 255)
COLOR_PALETTE_BG = (50, 50, 50)
COLOR_SELECTED = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BUTTON = (80, 80, 80)
COLOR_BUTTON_HOVER = (100, 100, 100)

# Terrain palette
TERRAIN_COLORS = {
    TerrainType.EMPTY: COLOR_EMPTY,
    TerrainType.BUILDING: COLOR_BUILDING,
    TerrainType.FIRE: COLOR_FIRE,
    TerrainType.SWAMP: COLOR_SWAMP,
    TerrainType.WATER: COLOR_WATER,
}

TERRAIN_NAMES = {
    TerrainType.EMPTY: "Empty",
    TerrainType.BUILDING: "Building",
    TerrainType.FIRE: "Fire",
    TerrainType.SWAMP: "Swamp",
    TerrainType.WATER: "Water",
}


class MapEditor:
    """Standalone map editor application."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Combat Environment - Map Editor")

        self.screen = pygame.display.set_mode((EDITOR_WIDTH, EDITOR_HEIGHT))
        self.clock = pygame.time.Clock()

        self.terrain_grid = TerrainGrid(GRID_SIZE, GRID_SIZE)
        self.selected_terrain = TerrainType.BUILDING
        self.show_grid = True
        self.running = True

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
        for action in ["Save", "Load", "Clear"]:
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
                self._handle_mouse_click(event)

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

        elif event.key == pygame.K_s:
            self._save_map()

        elif event.key == pygame.K_l:
            self._load_map()

        elif event.key == pygame.K_c:
            self.terrain_grid.clear()

        # Number keys for terrain selection
        elif event.key == pygame.K_1:
            self.selected_terrain = TerrainType.EMPTY
        elif event.key == pygame.K_2:
            self.selected_terrain = TerrainType.BUILDING
        elif event.key == pygame.K_3:
            self.selected_terrain = TerrainType.FIRE
        elif event.key == pygame.K_4:
            self.selected_terrain = TerrainType.SWAMP
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
                if action == "Save":
                    self._save_map()
                elif action == "Load":
                    self._load_map()
                elif action == "Clear":
                    self.terrain_grid.clear()
                return

        # Paint on grid
        if event.button == 1:  # Left click
            self._paint_at_position(mouse_pos, erase=False)
        elif event.button == 3:  # Right click
            self._paint_at_position(mouse_pos, erase=True)

    def _paint_at_position(self, mouse_pos, erase=False):
        """Paint terrain at mouse position."""
        x, y = mouse_pos

        # Check if in grid area
        if x < PALETTE_WIDTH:
            return

        # Convert to grid coordinates
        grid_x = (x - PALETTE_WIDTH) // CELL_SIZE
        grid_y = y // CELL_SIZE

        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            terrain = TerrainType.EMPTY if erase else self.selected_terrain
            self.terrain_grid.set(grid_x, grid_y, terrain)

    def _save_map(self):
        """Save the current map to a file."""
        if HAS_TKINTER:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Map"
            )
        else:
            # Use native OS dialog (macOS/Linux/Windows)
            filepath = _native_save_dialog()

        # Fallback to default if dialog was cancelled or unavailable
        if not filepath:
            filepath = "map.json"
            print(f"Using default filename: {filepath}")

        # Ensure .json extension
        if not filepath.endswith('.json'):
            filepath += '.json'
        save_map(self.terrain_grid, filepath)
        print(f"Map saved to: {filepath}")

    def _load_map(self):
        """Load a map from a file."""
        if HAS_TKINTER:
            filepath = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
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
            print(f"Map loaded from: {filepath}")
        except Exception as e:
            print(f"Error loading map: {e}")

    def _render(self):
        """Render the editor."""
        # Clear screen
        self.screen.fill(COLOR_BACKGROUND)

        # Render palette
        self._render_palette()

        # Render grid area
        self._render_terrain()

        if self.show_grid:
            self._render_grid_lines()

        # Render instructions
        self._render_instructions()

        pygame.display.flip()

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
        """Render the terrain grid."""
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                terrain = self.terrain_grid.get(x, y)
                color = TERRAIN_COLORS.get(terrain, COLOR_EMPTY)

                pixel_x = PALETTE_WIDTH + x * CELL_SIZE
                pixel_y = y * CELL_SIZE

                pygame.draw.rect(
                    self.screen,
                    color,
                    (pixel_x, pixel_y, CELL_SIZE, CELL_SIZE)
                )

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
        """Render keyboard shortcuts at bottom of palette."""
        instructions = [
            "1-5: Terrain",
            "S: Save",
            "L: Load",
            "C: Clear",
            "G: Grid",
            "Q: Quit",
        ]

        y = EDITOR_HEIGHT - len(instructions) * 18 - 10
        for line in instructions:
            text = self.small_font.render(line, True, (180, 180, 180))
            self.screen.blit(text, (10, y))
            y += 18


def main():
    """Entry point for the map editor."""
    editor = MapEditor()
    editor.run()


if __name__ == "__main__":
    main()
