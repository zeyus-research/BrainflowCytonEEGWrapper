"""Tests for EEG class using dummy board."""
import time
import numpy as np
import pytest
from BrainflowCyton.eeg import (
    EEG,
    CytonSampleRate,
    CytonChannel,
    CytonGain,
    CytonInputType,
    CytonCommand,
    Filtering,
    HAS_SOUNDDEVICE,
)


class TestEEGDummyBoard:
    """Test EEG class with dummy board."""

    def test_initialization(self):
        """Test EEG object initialization with dummy board."""
        eeg = EEG(dummyBoard=True)
        assert eeg.dummyBoard is True
        assert eeg.is_prepared is False
        assert len(eeg.exg_channels) > 0
        assert eeg.sampling_rate > 0

    def test_context_manager(self):
        """Test EEG as context manager."""
        with EEG(dummyBoard=True) as eeg:
            assert eeg.is_prepared is True
        # After exiting context, should be cleaned up
        assert eeg.is_prepared is False

    def test_start_stop_stream(self):
        """Test starting and stopping stream."""
        eeg = EEG(dummyBoard=True)
        eeg.start_stream(sdcard=False)
        assert eeg.is_prepared is True
        eeg.stop()
        assert eeg.is_prepared is False

    def test_poll_data(self):
        """Test polling data from dummy board."""
        with EEG(dummyBoard=True) as eeg:
            eeg.start_stream(sdcard=False)
            time.sleep(0.5)  # Let some data accumulate
            data = eeg.poll()
            assert data is not None
            assert isinstance(data, np.ndarray)
            assert data.shape[0] > 0  # Should have some channels

    def test_tag_insertion(self):
        """Test inserting markers/tags."""
        with EEG(dummyBoard=True) as eeg:
            eeg.start_stream(sdcard=False)
            time.sleep(0.2)
            # Should not raise exception
            eeg.tag('A')
            eeg.tag('1')


class TestEnums:
    """Test enum classes."""

    def test_sample_rate_conversion(self):
        """Test sample rate enum to Hz conversion."""
        assert CytonSampleRate.SR_250.to_hz() == 250
        assert CytonSampleRate.SR_1000.to_hz() == 1000
        assert CytonSampleRate.SR_2000.to_hz() == 2000

    def test_sample_rate_from_value(self):
        """Test getting sample rate from value."""
        assert CytonSampleRate.hz_from_value(6) == 250
        assert CytonSampleRate.hz_from_value(4) == 1000

    def test_channel_from_number(self):
        """Test getting channel from number."""
        ch1 = CytonChannel.from_channel_number(1)
        assert ch1 == CytonChannel.CH_1
        ch8 = CytonChannel.from_channel_number(8)
        assert ch8 == CytonChannel.CH_8

    def test_command_channel_on_off(self):
        """Test channel on/off commands."""
        ch1_on = CytonCommand.channel_number_on(1)
        assert ch1_on == CytonCommand.CHANNEL_1_ON
        ch1_off = CytonCommand.channel_number_off(1)
        assert ch1_off == CytonCommand.CHANNEL_1_OFF


class TestFiltering:
    """Test Filtering class."""

    def test_filtering_initialization(self):
        """Test Filtering object initialization."""
        ch_idx = [1, 2, 3, 4]
        filtering = Filtering(exg_channels=ch_idx, sampling_rate=250)
        assert filtering.exg_channels == ch_idx
        assert filtering.sampling_rate == 250

    def test_bandpass_filter(self):
        """Test bandpass filter on dummy data."""
        # Create dummy data: 5 channels x 1000 samples
        data = np.random.randn(5, 1000)
        ch_idx = [0, 1, 2, 3, 4]
        filtering = Filtering(exg_channels=ch_idx, sampling_rate=250)

        # Should not raise exception
        filtered = filtering.bandpass(data, lowcut=1.0, highcut=49.0)
        assert filtered.shape == data.shape

    def test_butterworth_lowpass(self):
        """Test butterworth lowpass filter."""
        data = np.random.randn(5, 1000)
        ch_idx = [0, 1, 2, 3, 4]
        filtering = Filtering(exg_channels=ch_idx, sampling_rate=250)

        filtered = filtering.butterworth_lowpass(data, cutoff=49.0)
        assert filtered.shape == data.shape


class TestLogging:
    """Test logging configuration."""

    def test_configure_logging(self):
        """Test logging configuration method."""
        import logging
        # Should not raise exception
        EEG.configure_logging(level=logging.DEBUG)
        EEG.configure_logging(level=logging.INFO)


class TestAudioOptional:
    """Test Audio class handles missing sounddevice gracefully."""

    def test_audio_import(self):
        """Test that Audio class can be imported even without sounddevice."""
        from BrainflowCyton.eeg import Audio, HAS_SOUNDDEVICE
        # Should be able to import regardless
        assert Audio is not None

    def test_audio_play_without_sounddevice(self):
        """Test that Audio.play raises helpful error when sounddevice missing."""
        from BrainflowCyton.eeg import Audio, HAS_SOUNDDEVICE
        import numpy as np

        if not HAS_SOUNDDEVICE:
            # Should raise ImportError with helpful message
            with pytest.raises(ImportError, match="sounddevice is required"):
                Audio.play(np.array([0.1, 0.2, 0.3]))
        else:
            # If sounddevice is installed, skip this specific test
            pytest.skip("sounddevice is installed")

    def test_audio_non_play_methods_work(self):
        """Test that Audio methods not requiring sounddevice still work."""
        from BrainflowCyton.eeg import Audio
        import numpy as np

        # These should work regardless of sounddevice
        data = np.random.randn(100)

        # Test smooth
        smoothed = Audio.smooth(data)
        assert smoothed.shape == data.shape

        # Test filter_savitzky_golay
        filtered = Audio.filter_savitzky_golay(data)
        assert filtered.shape == data.shape

        # Test resample
        resampled = Audio.resample(data, sr_in=250, sr_out=500)
        assert len(resampled) > 0
