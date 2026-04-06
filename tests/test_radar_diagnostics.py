import struct
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import radar


HEADER_STRUCT = struct.Struct("<4H8I")
TLV_STRUCT = struct.Struct("<2I")


def build_packet(tlvs, frame_number=1, num_detected_obj=0, subframe_number=0):
    payload = bytearray()
    for tlv_type, tlv_payload in tlvs:
        payload.extend(TLV_STRUCT.pack(tlv_type, len(tlv_payload)))
        payload.extend(tlv_payload)
    total_packet_len = radar.HEADER_SIZE + len(payload)
    header = HEADER_STRUCT.pack(
        0x0102,
        0x0304,
        0x0506,
        0x0708,
        0x01020304,
        total_packet_len,
        0,
        frame_number,
        0,
        num_detected_obj,
        len(tlvs),
        subframe_number,
    )
    return header + bytes(payload)


def build_short_track_record(
    track_id,
    *,
    x,
    y,
    z,
    vx=0.0,
    vy=0.0,
    vz=0.0,
    ax=0.0,
    ay=0.0,
    az=0.0,
    state=3.0,
    confidence=0.99,
    extras=(0.0, 0.0, 0.0, 0.0),
):
    return radar._TRACK_RECORD_SHORT_STRUCT.pack(
        int(track_id),
        float(x),
        float(y),
        float(z),
        float(vx),
        float(vy),
        float(vz),
        float(ax),
        float(ay),
        float(az),
        float(state),
        float(confidence),
        float(extras[0]),
        float(extras[1]),
        float(extras[2]),
        float(extras[3]),
    )


class ReplayLoadingTests(unittest.TestCase):
    def test_load_replay_capture_normalizes_frames(self):
        capture = radar.load_replay_capture(ROOT / "04_01_2026_16_12_52" / "replay_8.json")
        self.assertEqual(capture.demo, "3D People Tracking")
        self.assertEqual(capture.device, "xWR6843")
        self.assertEqual(len(capture.frames), 100)
        self.assertEqual(capture.schedule_s[0], 0.0)
        self.assertGreater(capture.schedule_s[-1], 0.0)
        self.assertGreater(capture.schedule_s[1], capture.schedule_s[0])

        first_frame = capture.frames[0]
        self.assertEqual(first_frame.frame_number, 701)
        self.assertEqual(first_frame.num_detected_obj, 86)
        self.assertEqual(len(first_frame.points), 86)
        self.assertEqual(len(first_frame.tracks), 2)
        self.assertTrue(all(track.bbox is not None for track in first_frame.tracks))
        self.assertEqual(capture.scene.sensor_height_m, 2.0)
        self.assertEqual(len(capture.scene.boundary_boxes), 1)
        self.assertEqual(len(capture.scene.presence_boxes), 1)

    def test_fusion_ready_payload_contains_tracks_and_scene(self):
        capture = radar.load_replay_capture(ROOT / "04_01_2026_16_12_52" / "replay_8.json")
        payload = capture.frames[1].fusion_ready_payload()
        self.assertEqual(payload["frame_number"], 702)
        self.assertEqual(payload["coordinate_frame"], "radar_sensor_xyz")
        self.assertEqual(len(payload["tracks"]), 2)
        self.assertIn("sensor_pose", payload["calibration"])


class PacketParsingTests(unittest.TestCase):
    def test_parse_compressed_point_packet(self):
        driver = radar.IWR6843Driver()
        units = struct.pack("<5f", 0.01, 0.01, 0.25, 0.1, 0.5)
        points = struct.pack("<bbHhH", 10, 4, 8, -3, 6) + struct.pack("<bbHhH", -12, 3, 10, 5, 8)
        packet = build_packet([(301, units + points)], frame_number=42, num_detected_obj=2)

        frame = driver.parse_packet_bytes(packet)

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(frame.frame_number, 42)
        self.assertEqual(len(frame.points), 2)
        self.assertAlmostEqual(frame.points[0].velocity, -0.3, places=4)
        self.assertGreater(frame.points[0].snr, 0.0)

    def test_parse_bad_header(self):
        driver = radar.IWR6843Driver()
        frame = driver.parse_packet_bytes(b"\x00" * 10)
        self.assertIsNone(frame)
        self.assertIn("header unpack failed", driver.session_state().last_parse_error)

    def test_parse_truncated_tlv(self):
        driver = radar.IWR6843Driver()
        payload = bytearray()
        payload.extend(TLV_STRUCT.pack(123, 16))
        payload.extend(b"\x01\x02")
        header = HEADER_STRUCT.pack(
            0x0102,
            0x0304,
            0x0506,
            0x0708,
            0x01020304,
            radar.HEADER_SIZE + len(payload),
            0,
            7,
            0,
            0,
            1,
            0,
        )
        frame = driver.parse_packet_bytes(header + payload)
        self.assertIsNotNone(frame)
        self.assertIn("bad TLV len", driver.session_state().last_parse_error)

    def test_unknown_tlv_is_recorded(self):
        driver = radar.IWR6843Driver()
        packet = build_packet([(999, b"abcde")], frame_number=5, num_detected_obj=0)
        frame = driver.parse_packet_bytes(packet)
        self.assertIsNotNone(frame)
        state = driver.session_state()
        self.assertEqual(state.unknown_tlv_counts.get(999), 1)
        self.assertEqual(state.unknown_tlv_lengths.get(999), [5])

    def test_parse_tracks_heights_indexes_and_presence(self):
        driver = radar.IWR6843Driver()

        units = struct.pack("<5f", 0.01, 0.01, 0.25, 0.1, 0.5)
        points = struct.pack("<bbHhH", 4, 2, 8, 0, 5) + struct.pack("<bbHhH", 5, 3, 9, 0, 6)
        packet_one = build_packet(
            [(radar.TLV_COMPRESSED_POINTS_LEGACY, units + points)],
            frame_number=1,
            num_detected_obj=2,
        )
        first_frame = driver.parse_packet_bytes(packet_one)
        self.assertIsNotNone(first_frame)

        track_payload = build_short_track_record(
            7,
            x=0.18,
            y=2.1,
            z=1.0,
            vx=0.05,
            confidence=0.98,
        )
        height_payload = radar._TRACK_HEIGHT_STRUCT.pack(7, 1.82, 0.12)
        index_payload = bytes([7, 7])
        presence_payload = struct.pack("<I", 1)
        packet_two = build_packet(
            [
                (radar.TLV_TRACK_LIST, track_payload),
                (radar.TLV_TRACK_HEIGHT, height_payload),
                (radar.TLV_TRACK_INDEX, index_payload),
                (radar.TLV_PRESENCE, presence_payload),
            ],
            frame_number=2,
            num_detected_obj=0,
        )

        frame = driver.parse_packet_bytes(packet_two)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(len(frame.tracks), 1)
        self.assertEqual(frame.tracks[0].track_id, 7)
        self.assertEqual(frame.presence, 1)
        self.assertEqual(frame.track_indexes, [7, 7])
        self.assertIsNotNone(frame.tracks[0].bbox)
        assert frame.tracks[0].bbox is not None
        self.assertAlmostEqual(frame.tracks[0].bbox.z_min, 0.12, places=4)
        self.assertAlmostEqual(frame.tracks[0].bbox.z_max, 1.82, places=4)
        self.assertEqual(frame.tracks[0].bbox_source, "associated_points")
        self.assertEqual(frame.calibration["camera_projection_hook"]["tracks_ready"], True)

    def test_project_box_to_camera(self):
        capture = radar.load_replay_capture(ROOT / "04_01_2026_16_12_52" / "replay_8.json")
        frame = capture.frames[1]
        camera = radar.CameraProjection()
        projection = frame.project_to_camera(camera)
        self.assertEqual(len(projection["tracks"]), len(frame.tracks))
        self.assertEqual(len(projection["points"]), len(frame.points))
        self.assertIsNotNone(projection["tracks"][0]["uv_bbox"])


class HealthVerdictTests(unittest.TestCase):
    def test_replay_mode_health(self):
        verdict, reason = radar.evaluate_health_verdict(
            mode="replay",
            config_ok=True,
            last_command_error="",
            connection_error="",
            rx_bytes=0,
            magic_hits=0,
            frames=0,
            last_parse_error="",
            data_opened_at=None,
            plots_updating=False,
        )
        self.assertEqual(verdict, radar.HEALTH_REPLAY_MODE)
        self.assertIn("hardware viability", reason)

    def test_config_failed_health(self):
        verdict, _ = radar.evaluate_health_verdict(
            mode="live",
            config_ok=False,
            last_command_error="sensorStart: timeout",
            connection_error="",
            rx_bytes=0,
            magic_hits=0,
            frames=0,
            last_parse_error="",
            data_opened_at=None,
            plots_updating=False,
        )
        self.assertEqual(verdict, radar.HEALTH_CONFIG_FAILED)

    def test_streaming_but_unparsed_health(self):
        verdict, reason = radar.evaluate_health_verdict(
            mode="live",
            config_ok=True,
            last_command_error="",
            connection_error="",
            rx_bytes=128,
            magic_hits=2,
            frames=0,
            last_parse_error="bad TLV len",
            data_opened_at=time.time() - 1.0,
            plots_updating=False,
        )
        self.assertEqual(verdict, radar.HEALTH_STREAMING_UNPARSED)
        self.assertIn("bad TLV len", reason)

    def test_no_data_health(self):
        verdict, reason = radar.evaluate_health_verdict(
            mode="live",
            config_ok=True,
            last_command_error="",
            connection_error="",
            rx_bytes=0,
            magic_hits=0,
            frames=0,
            last_parse_error="",
            data_opened_at=time.time() - 10.0,
            plots_updating=False,
        )
        self.assertEqual(verdict, radar.HEALTH_NO_DATA)
        self.assertIn("no data bytes", reason.lower())

    def test_healthy_verdict(self):
        verdict, reason = radar.evaluate_health_verdict(
            mode="live",
            config_ok=True,
            last_command_error="",
            connection_error="",
            rx_bytes=512,
            magic_hits=4,
            frames=3,
            last_parse_error="",
            data_opened_at=time.time() - 1.0,
            plots_updating=True,
        )
        self.assertEqual(verdict, radar.HEALTH_HEALTHY)
        self.assertIn("plots are updating", reason)


class PortDiscoveryTests(unittest.TestCase):
    def test_cp2105_pairing(self):
        infos = radar.discover_serial_ports(
            [
                SimpleNamespace(
                    device="/dev/ttyUSB0",
                    description="CP2105 Dual USB to UART Bridge Controller - Enhanced Com Port",
                    manufacturer="Silicon Labs",
                    product="CP2105 Dual USB to UART Bridge Controller",
                    interface="Enhanced Com Port",
                    hwid="USB VID:PID=10C4:EA70 SER=00ED228B LOCATION=1-2.3.2:1.0",
                ),
                SimpleNamespace(
                    device="/dev/ttyUSB1",
                    description="CP2105 Dual USB to UART Bridge Controller - Standard Com Port",
                    manufacturer="Silicon Labs",
                    product="CP2105 Dual USB to UART Bridge Controller",
                    interface="Standard Com Port",
                    hwid="USB VID:PID=10C4:EA70 SER=00ED228B LOCATION=1-2.3.2:1.1",
                ),
            ]
        )
        self.assertEqual(radar.suggest_serial_port_pairs(infos)[0], ("/dev/ttyUSB0", "/dev/ttyUSB1"))

    def test_xds110_pairing(self):
        infos = radar.discover_serial_ports(
            [
                SimpleNamespace(
                    device="/dev/ttyACM0",
                    description="XDS110 Class Application/User UART",
                    manufacturer="Texas Instruments",
                    product="XDS110",
                    interface="XDS110 Class Application/User UART",
                    hwid="USB VID:PID=0451:BEF3 SER=ABCD LOCATION=1-3:1.0 if00",
                ),
                SimpleNamespace(
                    device="/dev/ttyACM1",
                    description="XDS110 Class Auxiliary Data Port",
                    manufacturer="Texas Instruments",
                    product="XDS110",
                    interface="XDS110 Class Auxiliary Data Port",
                    hwid="USB VID:PID=0451:BEF3 SER=ABCD LOCATION=1-3:1.3 if03",
                ),
            ]
        )
        self.assertEqual(radar.suggest_serial_port_pairs(infos)[0], ("/dev/ttyACM0", "/dev/ttyACM1"))


if __name__ == "__main__":
    unittest.main()
