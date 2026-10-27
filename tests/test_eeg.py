"""Tests for EEG class using dummy and mocked boards."""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from BrainflowCyton.eeg import (
    EEG,
    HAS_SOUNDDEVICE,
    CytonChannel,
    CytonCommand,
    CytonGain,
    CytonInputType,
    CytonRecordingDuration,
    CytonSampleRate,
    Filtering,
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
            eeg.tag("A")
            eeg.tag("1")


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
        from BrainflowCyton.eeg import Audio

        # Should be able to import regardless
        assert Audio is not None

    def test_audio_play_without_sounddevice(self):
        """Test that Audio.play raises helpful error when sounddevice missing."""
        import numpy as np

        from BrainflowCyton.eeg import Audio

        if not HAS_SOUNDDEVICE:
            # Should raise ImportError with helpful message
            with pytest.raises(ImportError, match="sounddevice is required"):
                Audio.play(np.array([0.1, 0.2, 0.3]))
        else:
            # If sounddevice is installed, skip this specific test
            pytest.skip("sounddevice is installed")

    def test_audio_non_play_methods_work(self):
        """Test that Audio methods not requiring sounddevice still work."""
        import numpy as np

        from BrainflowCyton.eeg import Audio

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


def fake_board_response(command: str) -> str:
    """Responses a Cyton board gives to config_board commands (abridged)."""
    if command.startswith(CytonCommand.SAMPLE_RATE_PREFIX.value):
        return "Success: Sample rate is 1000Hz$$$"
    if command.startswith(CytonCommand.CHANNEL_CONFIG_PREFIX.value):
        return "Success: Channel set for 3$$$"
    if command in {d.value for d in CytonRecordingDuration}:
        return "Wiring is correct and a card is present.\nCorresponding SD file OBCI_01.TXT$$$"
    return ""


@pytest.fixture
def cyton():
    """A non-dummy EEG whose BoardShim is replaced with a mock, so no hardware is needed."""
    eeg = EEG(dummyBoard=False, serial_port="/dev/null", emg_channels=[3])
    eeg.board = MagicMock()
    eeg.board.config_board.side_effect = fake_board_response
    yield eeg
    eeg.is_prepared = False  # avoid __del__ calling stop() on the mock later


def sent_commands(eeg: EEG) -> list[str]:
    return [c.args[0] for c in eeg.board.config_board.call_args_list]


class TestEEGCytonCommands:
    """Test the commands sent to a (mocked) Cyton board."""

    def test_stream_over_dongle(self, cyton):
        cyton.start_stream(sdcard=False, sr=CytonSampleRate.SR_500)
        cyton.board.prepare_session.assert_called_once()
        cyton.board.start_stream.assert_called_once()
        assert sent_commands(cyton) == [
            "/0",  # default board mode
            "v",  # soft reset
            "d",  # channels to defaults
            "/4",  # marker mode
            "C",  # 16 channels
            f"x3{0}{CytonGain.GAIN_24.value}{CytonInputType.ADSINPUT_NORMAL.value}100X",  # EMG channel 3
            "~5",  # 500Hz
            "v",  # soft reset after sample rate change
        ]
        assert cyton.sampling_rate == 500
        assert cyton.sdcard is False

    def test_stream_over_dongle_without_markers(self, cyton):
        cyton.start_stream(sdcard=False, use_markers=False)
        assert "/4" not in sent_commands(cyton)
        assert "~6" in sent_commands(cyton)  # defaults to 250Hz

    def test_sd_recording_default_sample_rate(self, cyton):
        cyton.start_stream(sdcard=True, duration_max=60)
        commands = sent_commands(cyton)
        assert commands[-4:] == ["~4", "v", "K", "b"]  # 1000Hz, 12h block, start
        assert cyton.sdcard is True
        cyton.board.start_stream.assert_not_called()

    def test_sd_recording_custom_sample_rate(self, cyton):
        """A sample rate passed in SD card mode must actually be sent to the board."""
        cyton.start_stream(sdcard=True, sr=CytonSampleRate.SR_2000, duration_max=120)
        commands = sent_commands(cyton)
        assert commands[-4:] == ["~3", "v", "L", "b"]  # 2000Hz, 24h block, start
        assert "~4" not in commands
        assert cyton.sampling_rate == 2000

    def test_sd_recording_failure(self, cyton):
        cyton.board.config_board.side_effect = lambda c: (
            "sd card failed to initialize$$$" if c == "H" else fake_board_response(c)
        )
        with pytest.raises(Exception, match="Failed to start recording"):
            cyton.start_stream(sdcard=True, sr=CytonSampleRate.SR_250, duration_max=60)

    def test_stop_sd_recording(self, cyton):
        cyton.start_stream(sdcard=True)
        cyton.board.config_board.reset_mock()
        cyton.stop()
        assert sent_commands(cyton) == ["j", "s"]
        cyton.board.release_session.assert_called_once()
        cyton.board.stop_stream.assert_not_called()
        assert cyton.is_prepared is False

    def test_stop_stream(self, cyton):
        cyton.start_stream(sdcard=False)
        cyton.stop()
        cyton.board.stop_stream.assert_called_once()
        cyton.board.release_session.assert_called_once()
        cyton.stop()  # safe to call twice
        cyton.board.release_session.assert_called_once()

    def test_tag_sd_mode_sends_marker_command(self, cyton):
        cyton.start_stream(sdcard=True)
        cyton.tag("Z")
        assert sent_commands(cyton)[-1] == "`Z"
        cyton.board.insert_marker.assert_not_called()

    def test_tag_stream_mode_inserts_marker(self, cyton):
        cyton.start_stream(sdcard=False)
        cyton.tag("A")
        cyton.board.insert_marker.assert_called_once_with(float(ord("A")))

    @pytest.mark.parametrize("bad_tag", ["", "AB", 1])
    def test_tag_invalid(self, cyton, bad_tag):
        cyton.prepare()
        with pytest.raises(ValueError, match="single character"):
            cyton.tag(bad_tag)

    def test_requires_prepare(self, cyton):
        with pytest.raises(RuntimeError, match="not prepared"):
            cyton.poll()
        with pytest.raises(RuntimeError, match="not prepared"):
            cyton.tag("A")

    def test_poll_sd_mode_returns_none(self, cyton):
        cyton.start_stream(sdcard=True)
        assert cyton.poll() is None
        cyton.board.get_board_data.assert_not_called()

    def test_poll_clear(self, cyton):
        cyton.start_stream(sdcard=False)
        cyton.poll(clear=True)
        cyton.board.get_board_data.assert_called_once_with(cyton.num_points)
        cyton.poll(clear=False)
        cyton.board.get_current_board_data.assert_called_once_with(cyton.num_points)

    def test_enable_disable_channel(self, cyton):
        assert cyton.enable_channel(1) is True
        assert cyton.disable_channel(1) is True
        assert cyton.enable_channel(16) is True
        assert cyton.disable_channel(16) is True
        assert sent_commands(cyton) == ["!", "1", "I", "i"]

    @pytest.mark.parametrize("channel", [0, 17, -1])
    def test_channel_out_of_range(self, cyton, channel):
        with pytest.raises(ValueError, match="between 1 and 16"):
            cyton.enable_channel(channel)
        with pytest.raises(ValueError, match="between 1 and 16"):
            cyton.disable_channel(channel)

    def test_set_sample_rate_bad_response(self, cyton):
        cyton.board.config_board.side_effect = lambda c: "Failure: nope$$$"
        with pytest.raises(Exception, match="Unknown response"):
            cyton._set_sample_rate(CytonSampleRate.SR_1000)


class TestRecordingDuration:
    """Test SD card block selection."""

    @pytest.mark.parametrize(
        "minutes, sr, expected",
        [
            (1, CytonSampleRate.SR_250, CytonRecordingDuration.MIN_5),
            (10, CytonSampleRate.SR_250, CytonRecordingDuration.MIN_15),
            (60, CytonSampleRate.SR_250, CytonRecordingDuration.MIN_120),
            (60, CytonSampleRate.SR_1000, CytonRecordingDuration.MIN_720),
            (120, CytonSampleRate.SR_2000, CytonRecordingDuration.MIN_1440),
        ],
    )
    def test_smallest_block(self, minutes, sr, expected):
        assert CytonRecordingDuration.get_smallest_block_for_time(minutes, sr) is expected

    def test_too_long(self):
        with pytest.raises(ValueError, match="1000Hz sampling rate"):
            CytonRecordingDuration.get_smallest_block_for_time(24 * 60, CytonSampleRate.SR_1000)


class TestFilteringExtra:
    """Filtering methods not covered above."""

    def test_filter_50hz(self):
        sr = 250
        t = np.arange(sr * 4) / sr
        data = np.zeros((2, len(t)))
        data[1] = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t)
        filtered = Filtering(exg_channels=[1], sampling_rate=sr).filter_50hz(data.copy())

        spectrum = np.abs(np.fft.rfft(filtered[1][sr:-sr]))
        freqs = np.fft.rfftfreq(len(t) - 2 * sr, 1 / sr)
        assert spectrum[freqs == 50] < 0.1 * spectrum[freqs == 10]

    def test_resample(self):
        data = np.random.default_rng(0).standard_normal((1000, 2))  # (samples, channels)
        resampled = Filtering(exg_channels=[0, 1], sampling_rate=250).resample(data, 500)
        assert resampled.shape == (2000, 2)
