# Changelog

## [0.2.1] - 2025-10-20

### Fixed
- Fixed github runners to include portaudio

## [0.2.0] - 2025-10-20

### Added
- Context manager support for EEG class (use with `with` statement)
- Static method `EEG.configure_logging()` for easier logging setup
- Comprehensive test suite with pytest
- PEP 561 compatibility with py.typed marker file
- CHANGELOG.md for tracking project changes

### Changed
- Migrated from `nptyping` to `numpy.typing` for better type hint support
- Removed `nptyping` dependency (using modern numpy.typing instead)
- Fixed static method decorators in Audio, EEGReader, and EEGWriter classes
- Fixed fragile `__class__` reference in CytonSampleRate enum
- Updated README examples with corrected code patterns
- Fixed typo: "Full functional" → "Fully functional"
- Fixed checkbox syntax in README todo list
- Improved .gitignore with more comprehensive Python patterns

### Fixed
- Missing `@staticmethod` decorators on utility methods
- Version mismatch between pyproject.toml and __init__.py
- Bandpass filter example in README (moved start_stream out of loop)
- Linter warnings for unused parameters

## [0.1.0] - Initial Release

### Added
- Initial implementation of BrainFlow wrapper for Cyton/Cyton + Daisy
- Support for custom sample rates
- Channel configuration capabilities
- SD card recording support
- Real-time streaming over dongle
- Filtering utilities (bandpass, lowpass, 50Hz filter)
- Audio utilities for EEG sonification
- EEG file readers/writers (OpenBCI TXT, XDF formats)
- Marker/tag support for event marking
- EMG channel configuration
- Dummy board support for testing
