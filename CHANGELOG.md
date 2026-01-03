# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## v0.1.1

### Added
- `map_editor.py` - Standalone map editor for creating custom terrain maps
- `combatenv/map_io.py` - Save/load functions for JSON map files (`save_map`, `load_map`)
- `tests/test_map_io.py` - Tests for map I/O functionality
- `env.reset(options={'terrain_grid': terrain})` - Support for custom terrain in environment reset

## v0.1.0

### Added
- Configuration validation for terrain percentages (warns if total exceeds 50%)
- FOV cache cleanup when agents die (memory optimization)

### Changed
- `SpatialGrid.get_nearby_agents()` no longer accepts unused `radius` parameter

### Removed
- `InputHandler` class and `input_handler.py` module (duplicate of environment's event handling)
- `tests/test_input_handler.py` test file

### Fixed
- FOV cache memory leak when agents died without cache cleanup
