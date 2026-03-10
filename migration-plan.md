# Plan: Migrate from pygame to pyglet

## Context

Students on university Windows machines cannot install combatenv because pygame requires C/C++ compilation and they lack admin privileges. Even with up-to-date pip, the pre-built wheels fail. **pyglet** is a pure-Python library (uses ctypes to OpenGL) that installs with a simple `pip install` — no compiler, no admin rights.

## Why pyglet (not arcade)

- **Pure Python** — zero C/C++ compilation
- **Lightweight** — no transitive compiled dependencies (arcade pulls in pyglet + pillow + pymunk)
- **Good 2D primitives** — `pyglet.shapes` (Circle, Line, Rectangle) + batch rendering
- **numpy → texture** — `pyglet.image.ImageData` for terrain LUT rendering
- **Active development** — pyglet 2.x

## Scope

**Files that need changes (7 + tests + config):**

| File | Lines | Change |
|------|-------|--------|
| `combatenv/renderer.py` | ~1645 | Full rewrite of drawing calls |
| `combatenv/environment.py` | ~1388 | Init, display, events, clock, rgb_array |
| `combatenv/wrappers/render_wrapper.py` | ~465 | Surface, draw, events, display |
| `combatenv/wrappers/keybindings.py` | ~271 | Event types, key constants |
| `combatenv/gridworld.py` | ~191 | Init, display, clock |
| `combatenv/map_editor.py` | ~621 | Standalone app rewrite |
| `combatenv/preview_terrain.py` | ~200 | Standalone app rewrite |
| `tests/test_renderer.py` | ~varies | Surface → RenderContext in fixtures |
| `pyproject.toml` / `requirements.txt` | — | `pygame` → `pyglet>=2.0` |

**Files unchanged (zero pygame imports):** `agent.py`, `fov.py`, `spatial.py`, `terrain.py`, `projectile.py`, `config.py`, `map_io.py`, `boids.py`, `unit.py`, all other wrappers, all other tests.

## Key Paradigm Differences

### 1. No "Surface" concept
pygame draws to `Surface` objects then blits them together. pyglet draws shapes/textures directly to the OpenGL framebuffer, layered via `OrderedGroup`.

**Solution:** Replace `surface: pygame.Surface` parameter with a `RenderContext` that holds a `pyglet.graphics.Batch` and ordered groups for layering.

### 2. surfarray → texture upload
`pygame.surfarray.blit_array(surface, numpy_array)` becomes:
```python
image_data = pyglet.image.ImageData(w, h, 'RGBA', rgba.tobytes(), pitch=-w*4)
sprite = pyglet.sprite.Sprite(image_data.get_texture(), batch=batch, group=group)
```
Used in: terrain LUT rendering, water/forest/lava/mountain variations, FOV overlays.

### 3. Event model
pygame polls events (`pygame.event.get()` returns list). pyglet dispatches callbacks.

**Solution:** `EventCollector` class that registers pyglet callbacks, queues events, then provides a `poll()` method matching the existing loop structure.

### 4. Y-axis flip
pygame: (0,0) top-left, Y down. pyglet: (0,0) bottom-left, Y up.

**Solution:** Helper `flip_y(y) → WINDOW_SIZE - y` used in all coordinate conversions.

### 5. Key constants
`pygame.K_q` → `pyglet.window.key.Q`, `pygame.KMOD_SHIFT` → `pyglet.window.key.MOD_SHIFT`, etc.

## Implementation Phases

### Phase 0: Infrastructure (`combatenv/graphics.py` — new file)

Create the bridge module with:
- **`RenderContext`** — wraps `pyglet.window.Window`, `pyglet.graphics.Batch`, and `OrderedGroup`s for each layer (background, terrain, grid, fov, agents, projectiles, flash, overlay)
- **`EventCollector`** — registers `on_key_press`, `on_mouse_press`, `on_close`, `on_mouse_motion` callbacks, provides `poll()` returning event list
- **`flip_y()`** — coordinate transform helper
- **Texture helpers** — `numpy_to_texture(rgba_array)` for terrain/overlay rendering

### Phase 1: Migrate `renderer.py` (highest risk, do first)

Everything else depends on this. Replace in order:

1. **FOV overlay surfaces** (7 pre-allocated `pygame.Surface` with `SRCALPHA`) → pre-computed RGBA numpy arrays, uploaded as small textures
2. **`render_background()`** → `pyglet.shapes.Rectangle` in background group
3. **`render_grid_lines()`** → list of `pyglet.shapes.Line` in grid group (cached)
4. **`render_terrain()`** → build RGB LUT array (same numpy code), add alpha channel, upload as texture via `pyglet.image.ImageData`
5. **Terrain variations** (`render_water_depth`, etc.) → build RGBA arrays in numpy (cleaner than current `pixels3d`/`pixels_alpha`), upload as textures
6. **`render_fov_highlights()`** → `pyglet.shapes.Rectangle` with RGBA color per cell, all in one batch
7. **`render_agent()`** → `pyglet.shapes.Circle` + `pyglet.shapes.Line` (nose)
8. **`render_projectile()` / `render_muzzle_flash()`** → pyglet shapes
9. **`render_debug_overlay()` / `render_keybindings_overlay()`** → `pyglet.shapes.Rectangle` background + `pyglet.text.Label` for text
10. **`render_waypoints()` / `render_selected_unit()`** → pyglet shapes
11. **`render_all()`** → accept `RenderContext`, call batch.draw() at end

Change all function signatures: `surface: pygame.Surface` → `ctx: RenderContext`.

### Phase 2: Migrate `environment.py`

1. Replace `self._screen`, `self._clock`, `self._pygame_initialized` with `self._window`, `self._events`, `self._render_ctx`
2. `_init_pygame()` → `_init_graphics()`: create `pyglet.window.Window`, `EventCollector`, `RenderContext`
3. `process_events()` → use `window.dispatch_events()` + `self._events.poll()`, map key constants
4. `render()` → call `render_all(self._render_ctx, ...)`, then `window.flip()`
5. `rgb_array` mode → `pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()` → numpy array
6. `close()` → `window.close()`
7. Timing → `time.perf_counter()` based (simpler than pyglet.clock for this use case)

### Phase 3: Migrate `gridworld.py`

Same pattern as environment.py but simpler (only ~5 pygame calls). Lazy init with pyglet window.

### Phase 4: Migrate `wrappers/keybindings.py`

- Change `PYGAME_AVAILABLE` → `PYGLET_AVAILABLE` (same graceful degradation pattern)
- Replace all `pygame.K_*` → `pyglet.window.key.*`
- Replace event type checks to match `EventCollector` tuple format

### Phase 5: Migrate `wrappers/render_wrapper.py`

- Replace `pygame.Surface` muzzle flash → pyglet shapes
- Replace `pygame.display.flip()` → `window.flip()`
- Replace clock with perf_counter timing
- Update all `render_*` calls to pass `RenderContext`

### Phase 6: Migrate standalone tools

- **`preview_terrain.py`**: Small app, straightforward port
- **`map_editor.py`**: Larger but standalone; replace mouse tracking with `EventCollector`, all drawing with pyglet shapes

### Phase 7: Update tests & dependencies

- `tests/test_renderer.py`: Replace `pygame.Surface` fixtures with `RenderContext`
- `pyproject.toml`: `"pygame"` → `"pyglet>=2.0"`
- `requirements.txt`: `pygame>=2.0.0` → `pyglet>=2.0`
- Run full test suite to confirm simulation tests pass unchanged

## Verification

1. **Unit tests**: `python -m pytest tests/ -v` — all 193+ tests pass
2. **Visual smoke test**: `python main.py` — window opens, 200 agents render, keyboard controls work
3. **rgb_array mode**: verify `env.render()` returns `(H, W, 3)` numpy array
4. **Headless mode**: `TacticalCombatEnv(render_mode=None)` runs without errors
5. **Map editor**: `python -m combatenv.map_editor` — paints terrain, saves/loads
6. **FPS benchmark**: confirm ≥50 FPS with 200 agents (baseline: ~58 FPS with pygame)
7. **Install test**: `pip install pyglet` on a clean Windows Python 3.10+ machine without admin — no compilation errors
