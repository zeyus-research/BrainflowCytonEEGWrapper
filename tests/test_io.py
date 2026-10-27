"""Tests for EEGReader / EEGWriter file I/O."""

import numpy as np
import pytest

from BrainflowCyton.eeg import EEGReader, EEGWriter


def make_stream(name: str, channel_format: str, time_series, time_stamps, srate: float = 250.0) -> dict:
    n_channels = len(time_series[0]) if len(time_series) else 1
    return {
        "info": {
            "name": [name],
            "type": ["EEG" if channel_format != "string" else "Markers"],
            "channel_count": [str(n_channels)],
            "nominal_srate": [str(srate)],
            "channel_format": [channel_format],
            "source_id": [f"{name}-source"],
            "desc": [None],
        },
        "time_series": time_series,
        "time_stamps": np.asarray(time_stamps, dtype=np.float64),
    }


def read_xdf_raw(path) -> tuple[list[dict], dict]:
    """Read without clock sync / dejitter so timestamps come back exactly as written."""
    import pyxdf

    return pyxdf.load_xdf(str(path), synchronize_clocks=False, dejitter_timestamps=False)


class TestXDF:
    """Test XDF writing and reading."""

    @pytest.mark.parametrize(
        "channel_format, dtype",
        [
            ("float32", np.float32),
            ("double64", np.float64),
            ("int8", np.int8),
            ("int16", np.int16),
            ("int32", np.int32),
            ("int64", np.int64),
        ],
    )
    def test_numeric_roundtrip(self, tmp_path, channel_format, dtype):
        """Numeric streams round-trip with exact values and timestamps."""
        rng = np.random.default_rng(0)
        data = (rng.standard_normal((300, 4)) * 100).astype(dtype)
        stamps = 1000.0 + np.arange(300) / 250.0
        path = tmp_path / "numeric.xdf"

        EEGWriter.write_xdf(str(path), [make_stream("EEG", channel_format, data, stamps)])
        streams, header = read_xdf_raw(path)

        assert header["info"]["version"] == ["1.0"]
        assert len(streams) == 1
        stream = streams[0]
        assert stream["info"]["name"] == ["EEG"]
        assert stream["info"]["channel_count"] == ["4"]
        assert stream["info"]["channel_format"] == [channel_format]
        assert stream["time_series"].dtype == dtype
        np.testing.assert_array_equal(stream["time_series"], data)
        np.testing.assert_array_equal(stream["time_stamps"], stamps)

    def test_string_marker_stream(self, tmp_path):
        """String streams (e.g. markers) round-trip, including non-ASCII text."""
        markers = [["start"], ["stimulus ✓"], ["end"]]
        stamps = [1.5, 2.25, 10.0]
        path = tmp_path / "markers.xdf"

        EEGWriter.write_xdf(str(path), [make_stream("Markers", "string", markers, stamps, srate=0)])
        streams, _ = read_xdf_raw(path)

        assert streams[0]["time_series"] == markers
        np.testing.assert_array_equal(streams[0]["time_stamps"], stamps)

    def test_multiple_streams_and_metadata(self, tmp_path):
        """Multiple streams keep their order, and nested desc metadata survives."""
        eeg = make_stream("EEG", "float32", np.ones((10, 2), dtype=np.float32), np.arange(10) / 250.0)
        eeg["info"]["desc"] = [
            {"channels": [{"channel": [{"label": ["Fp1"], "unit": ["uV"]}, {"label": ["Fp2"], "unit": ["uV"]}]}]}
        ]
        markers = make_stream("Markers", "string", [["a"]], [0.5], srate=0)
        path = tmp_path / "multi.xdf"

        EEGWriter.write_xdf(str(path), [eeg, markers])
        streams, _ = read_xdf_raw(path)

        assert [s["info"]["name"][0] for s in streams] == ["EEG", "Markers"]
        channels = streams[0]["info"]["desc"][0]["channels"][0]["channel"]
        assert [c["label"][0] for c in channels] == ["Fp1", "Fp2"]
        assert streams[0]["footer"]["info"]["sample_count"] == ["10"]

    def test_empty_stream(self, tmp_path):
        """A stream with no samples is written as header + footer only."""
        path = tmp_path / "empty.xdf"
        stream = make_stream("Empty", "float32", np.zeros((0, 3), dtype=np.float32), [])
        stream["info"]["channel_count"] = ["3"]

        EEGWriter.write_xdf(str(path), [stream])
        streams, _ = read_xdf_raw(path)

        assert streams[0]["info"]["name"] == ["Empty"]
        assert len(streams[0]["time_stamps"]) == 0

    def test_large_chunk_length(self, tmp_path):
        """Chunks longer than 255 bytes use the 4-byte length encoding."""
        data = np.arange(20000 * 8, dtype=np.float64).reshape(20000, 8)
        stamps = np.arange(20000) / 1000.0
        path = tmp_path / "large.xdf"

        EEGWriter.write_xdf(str(path), [make_stream("EEG", "double64", data, stamps, srate=1000)])
        streams, _ = read_xdf_raw(path)

        np.testing.assert_array_equal(streams[0]["time_series"], data)

    def test_read_write_read_roundtrip(self, tmp_path):
        """Output of EEGReader.read_xdf can be written straight back with EEGWriter.write_xdf."""
        data = np.random.default_rng(1).standard_normal((500, 3)).astype(np.float32)
        stamps = 50.0 + np.arange(500) / 250.0
        first = tmp_path / "first.xdf"
        second = tmp_path / "second.xdf"
        EEGWriter.write_xdf(
            str(first),
            [make_stream("EEG", "float32", data, stamps), make_stream("Markers", "string", [["x"]], [51.0], 0)],
        )

        streams, header = EEGReader.read_xdf(str(first))
        EEGWriter.write_xdf(str(second), streams, header)
        streams2, header2 = EEGReader.read_xdf(str(second))

        assert header2 == header
        for s1, s2 in zip(streams, streams2, strict=True):
            for key in ("name", "type", "channel_count", "nominal_srate", "channel_format", "source_id"):
                assert s2["info"][key] == s1["info"][key]
            np.testing.assert_allclose(s2["time_stamps"], s1["time_stamps"])
        np.testing.assert_array_equal(streams2[0]["time_series"], data)
        assert streams2[1]["time_series"] == [["x"]]

    def test_default_header(self, tmp_path):
        """headers is optional and defaults to an XDF 1.0 file header."""
        path = tmp_path / "noheader.xdf"
        EEGWriter.write_xdf(str(path), [make_stream("EEG", "float32", np.zeros((2, 1)), [0.0, 0.004])])
        _, header = read_xdf_raw(path)
        assert header == {"info": {"version": ["1.0"]}}


class TestOpenBCITxt:
    """Test reading OpenBCI GUI .txt recordings."""

    @pytest.fixture
    def obci_file(self, tmp_path):
        path = tmp_path / "OpenBCI-RAW.txt"
        path.write_text(
            "%OpenBCI Raw EEG Data\n"
            "%Number of channels = 16\n"
            "%Sample Rate = 125 Hz\n"
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
            "Sample Index, EXG Channel 0, EXG Channel 1, Timestamp\n"
            "0, 1.5, -2.5, 1700000000.0\n"
            "1, 3.0, -4.0, 1700000000.008\n"
            "2, 4.5, -5.5, 1700000000.016\n"
        )
        return str(path)

    def test_parse_header(self, obci_file):
        headers, skip = EEGReader.parse_obci_header(obci_file)
        assert skip == 4
        assert headers == {
            "exg_channels": 16,
            "sampling_rate": 125,
            "board": "OpenBCI_GUI$BoardCytonSerialDaisy",
        }

    def test_read_txt(self, obci_file):
        df, headers = EEGReader.read_openbci_txt(obci_file)
        assert headers["sampling_rate"] == 125
        assert len(df) == 3
        assert list(df.columns) == ["Sample Index", " EXG Channel 0", " EXG Channel 1", " Timestamp"]
        np.testing.assert_array_equal(df[" EXG Channel 0"].to_numpy(), [1.5, 3.0, 4.5])
