# BrainflowCytonEEGWrapper

[![Test](https://github.com/zeyus-research/BrainflowCytonEEGWrapper/actions/workflows/test.yml/badge.svg)](https://github.com/zeyus-research/BrainflowCytonEEGWrapper/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/BrainflowCyton.svg)](https://badge.fury.io/py/BrainflowCyton)
[![Python versions](https://img.shields.io/pypi/pyversions/BrainflowCyton.svg)](https://pypi.org/project/BrainflowCyton/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is particularly aimed at the OpenBCI Cyton/Cyton + Daisy biosensing board.

It allows for low-level board commands such as configuring channels and setting sample rate, making it easier to configure the way you retrieve data from the board.

## Status

Fully functional, working with Cyton and Cyton + Daisy.

Todo:
- [ ] Include script to process SDCard data (exists but in a different repo)
- [ ] Make the interface more consistent
- [ ] Bake in multithreading support for easier parallel data collection + processing
- [ ] ???

## Details

Default channels coming in with cyton and daisy with the ultracortex mk4 are:

- 0: "pkg"
- 1: "Fp1"
- 2: "Fp2"
- 3: "C3"
- 4: "C4"
- 5: "P7"
- 6: "P8"
- 7: "O1"
- 8: "O2"
- 9: "F7"
- 10: "F8"
- 11: "F3"
- 12: "F4"
- 13: "T7"
- 14: "T8"
- 15: "P3"
- 16: "P4"
- 17: "AX" (accelerometer x)
- 18: "AY" (accelerometer y)
- 19: "AZ" (accelerometer z)
- 31: "marker" (this can be used to put event markers in the EEG data, which is extremely useful, BUT the accelerometer will be disabled)

## Installation

This package requires portaudio to be installed on your system.

On macOS, you can install it via Homebrew:

```bash
brew install portaudio
```

On Ubuntu/Debian, you can install it via apt:

```bash
sudo apt-get install portaudio19-dev
```

On Windows, it should be installed alongside the package.

Then, you can install the package via pip:

```bash
pip install BrainflowCyton
```

Or for development:

```bash
git clone https://github.com/zeyus-research/BrainflowCytonEEGWrapper
cd BrainflowCytonEEGWrapper
pip install -e .
```

## Features

- Low-level control of OpenBCI Cyton/Cyton + Daisy boards
- Support for custom sample rates (250Hz - 16kHz)
- SD card recording and real-time streaming
- Channel configuration (gain, input type, bias, SRB settings)
- EMG channel support
- Event markers/tags for experiment synchronization
- Built-in filtering (bandpass, lowpass, 50Hz noise removal)
- Context manager support for automatic resource cleanup
- Type hints for better IDE support
- Comprehensive test suite

## Usage examples

### Read data from a dummy board in real time (recommended pattern)

Using context manager (automatically handles cleanup):

```python
from BrainflowCyton.eeg import EEG
from time import sleep

# Context manager automatically calls prepare() and stop()
with EEG(dummyBoard=True) as eeg:
    eeg.start_stream(sdcard=False)
    while True:
        try:
            sleep(0.5)
            data = eeg.poll()
            print(f"Got {data.shape[1]} samples")
        except KeyboardInterrupt:
            break
```

Or manually managing the connection:

```python
from BrainflowCyton.eeg import EEG
from time import sleep

eeg = EEG(dummyBoard=True)
eeg.start_stream(sdcard=False)

while True:
    try:
        sleep(0.5)
        data = eeg.poll()
    except KeyboardInterrupt:
        eeg.stop()
        break
```

### Read data from a real board in real time

```python
from BrainflowCyton.eeg import EEG
from time import sleep

eeg_source = EEG()
eeg_source.start_stream(sdcard = False)

while True:
  try:
    sleep(0.5)
    data = eeg_source.poll()
  except KeyboardInterrupt:
    eeg_source.stop()
  
```

### Set a custom sample rate

*Note: to use sample rates above 250, an SDCard is required, streaming is limited to 250 Hz.*

```python
from BrainflowCyton.eeg import EEG, CytonSampleRate
from time import sleep

eeg_source = EEG()
eeg_source.start_stream(sdcard = True, sr = CytonSampleRate.SR_1000)

while True:
  try:
    sleep(0.5)
    data = eeg_source.poll()
  except KeyboardInterrupt:
    eeg_source.stop()
  
```

### Enable logging

```python
from BrainflowCyton.eeg import EEG
import logging

# Configure logging before creating EEG object
EEG.configure_logging(level=logging.DEBUG)

with EEG(dummyBoard=True) as eeg:
    eeg.start_stream(sdcard=False)
    # Will now see detailed logs
```

### Bandpass the data

```python
from BrainflowCyton.eeg import EEG, Filtering
from time import sleep

# Set the indexes of channels you want to filter
ch_idx = [1, 2, 3, 4, 5, 6, 7]
eeg_filter = Filtering(exg_channels=ch_idx, sampling_rate=250)

with EEG(dummyBoard=True) as eeg:
    eeg.start_stream(sdcard=False)
    while True:
        try:
            sleep(0.5)
            data = eeg.poll()
            if data is not None:
                # Apply 8-32 Hz bandpass (alpha + beta band)
                filtered_data = eeg_filter.bandpass(data, lowcut=8, highcut=32)
        except KeyboardInterrupt:
            break
```

## Development

### Running tests

```bash
# Install dev dependencies
uv sync --all-extras --dev

# Run tests
uv run pytest tests/ -v
```

### Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

## CI/CD

This project uses GitHub Actions for continuous integration and deployment:

- **Tests**: Run on every push/PR across Python 3.9-3.13 and Linux/macOS/Windows
- **Pre-releases**: Automatically published to GitHub releases on main branch commits
- **Releases**: Triggered by version tags (e.g., `v0.3.0`), publishes to PyPI

## License

MIT License - see LICENSE file for details
