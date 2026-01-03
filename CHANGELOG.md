# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
