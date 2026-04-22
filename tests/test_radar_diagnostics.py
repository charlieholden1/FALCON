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


class FakeSerialPort:
    def __init__(
        self,
        response: bytes = b"sensorStop\nDone\n",
        *,
        name: str = "serial",
        events=None,
    ):
        self._read_buffer = bytearray(response)
        self.name = name
        self.events = events if events is not None else []
        self.writes = []
        self.is_open = True

    @property
    def in_waiting(self):
        return len(self._read_buffer)

    def write(self, data):
        self.writes.append(bytes(data))
        self.events.append((self.name, "write", bytes(data)))
        return len(data)

    def read(self, size=1):
        if not self._read_buffer:
            return b""
        size = max(1, int(size))
        chunk = bytes(self._read_buffer[:size])
        del self._read_buffer[:size]
        return chunk

    def reset_input_buffer(self):
        self._read_buffer.clear()

    def close(self):
        self.events.append((self.name, "close", b""))
        self.is_open = False


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
        self.assertAlmostEqual(frame.points[0].x, 0.1995, places=3)
        self.assertAlmostEqual(frame.points[0].y, 1.9884, places=3)
        self.assertAlmostEqual(frame.points[0].z, 0.0800, places=3)
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
        self.assertEqual(frame.tracks[0].bbox_source, "associated_points_clamped")
        self.assertEqual(frame.calibration["camera_projection_hook"]["tracks_ready"], True)

    def test_parse_3d_people_counting_tlv_aliases_with_multiple_tracks(self):
        driver = radar.IWR6843Driver()

        units = struct.pack("<5f", 0.01, 0.01, 0.25, 0.1, 0.5)
        points = b"".join(
            [
                struct.pack("<bbHhH", 0, 0, 8, 0, 5),
                struct.pack("<bbHhH", 2, 0, 9, 0, 5),
                struct.pack("<bbHhH", 10, 1, 12, 0, 8),
                struct.pack("<bbHhH", 12, 1, 13, 0, 8),
            ]
        )
        track_payload = b"".join(
            [
                build_short_track_record(1, x=0.02, y=2.0, z=1.0, confidence=0.95),
                build_short_track_record(2, x=0.45, y=3.0, z=1.1, confidence=0.93),
            ]
        )
        height_payload = radar._TRACK_HEIGHT_STRUCT.pack(1, 1.78, 0.10)
        height_payload += radar._TRACK_HEIGHT_STRUCT.pack(2, 1.86, 0.12)
        index_payload = bytes([1, 1, 2, 2])
        presence_payload = struct.pack("<I", 1)

        packet = build_packet(
            [
                (radar.TLV_3D_PEOPLE_COMPRESSED_POINTS, units + points),
                (radar.TLV_3D_PEOPLE_TRACK_LIST, track_payload),
                (radar.TLV_3D_PEOPLE_TRACK_HEIGHT, height_payload),
                (radar.TLV_3D_PEOPLE_TRACK_INDEX, index_payload),
                (radar.TLV_3D_PEOPLE_PRESENCE, presence_payload),
            ],
            frame_number=7,
            num_detected_obj=4,
        )

        frame = driver.parse_packet_bytes(packet)
        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(len(frame.points), 4)
        self.assertEqual(len(frame.tracks), 2)
        self.assertEqual([track.track_id for track in frame.tracks], [1, 2])
        self.assertEqual(frame.track_indexes, [1, 1, 2, 2])
        self.assertEqual(frame.presence, 1)
        self.assertNotIn(radar.TLV_3D_PEOPLE_COMPRESSED_POINTS, driver.session_state().unknown_tlv_counts)
        self.assertNotIn(radar.TLV_3D_PEOPLE_TRACK_INDEX, driver.session_state().unknown_tlv_counts)
        for track in frame.tracks:
            self.assertIsNotNone(track.bbox)
            assert track.bbox is not None
            self.assertLessEqual(track.bbox.width, radar.PERSON_BOX_MAX_WIDTH_M + 1e-9)
            self.assertLessEqual(track.bbox.depth, radar.PERSON_BOX_MAX_DEPTH_M + 1e-9)

    def test_associated_points_do_not_make_giant_person_boxes(self):
        track = radar.RadarTrack(track_id=1, x=0.0, y=2.0, z=1.0)
        points = [
            radar.RadarPoint(x=-5.0, y=0.5, z=0.2, velocity=0.0),
            radar.RadarPoint(x=0.0, y=2.0, z=1.0, velocity=0.0),
            radar.RadarPoint(x=5.0, y=6.0, z=2.5, velocity=0.0),
        ]
        bbox, source = radar._build_track_bbox(
            track,
            associated_points=points,
            height_range=None,
        )
        self.assertEqual(source, "associated_points_clamped")
        self.assertLessEqual(bbox.width, radar.PERSON_BOX_MAX_WIDTH_M + 1e-9)
        self.assertLessEqual(bbox.depth, radar.PERSON_BOX_MAX_DEPTH_M + 1e-9)
        self.assertLessEqual(bbox.height, radar.PERSON_BOX_MAX_HEIGHT_M + 1e-9)

    def test_project_box_to_camera(self):
        capture = radar.load_replay_capture(ROOT / "04_01_2026_16_12_52" / "replay_8.json")
        frame = capture.frames[1]
        camera = radar.CameraProjection()
        projection = frame.project_to_camera(camera)
        self.assertEqual(len(projection["tracks"]), len(frame.tracks))
        self.assertEqual(len(projection["points"]), len(frame.points))
        self.assertIsNotNone(projection["tracks"][0]["uv_bbox"])

    def test_frame_debug_payload_summarizes_tracks_and_points(self):
        bbox = radar.RadarBox3D(
            x_min=-0.3,
            x_max=0.3,
            y_min=1.7,
            y_max=2.3,
            z_min=0.0,
            z_max=1.8,
            track_id=4,
        )
        frame = radar.RadarFrame(
            frame_number=10,
            subframe_number=0,
            num_detected_obj=2,
            num_tlvs=4,
            points=[
                radar.RadarPoint(x=-0.1, y=2.0, z=0.5, velocity=-0.2, snr=8.0),
                radar.RadarPoint(x=0.1, y=2.2, z=1.0, velocity=0.3, snr=10.0),
            ],
            timestamp=123.0,
            tracks=[
                radar.RadarTrack(
                    track_id=4,
                    x=0.0,
                    y=2.0,
                    z=1.0,
                    confidence=0.9,
                    bbox=bbox,
                    associated_point_indexes=[0, 1],
                    bbox_source="test",
                )
            ],
            track_indexes=[4, 4],
            presence=1,
        )

        payload = radar.radar_frame_debug_payload(frame, include_points=True)
        self.assertEqual(payload["frame_number"], 10)
        self.assertEqual(payload["point_count"], 2)
        self.assertEqual(payload["track_count"], 1)
        self.assertEqual(payload["track_index_counts"], {"4": 2})
        self.assertAlmostEqual(payload["tracks"][0]["bbox"]["height"], 1.8)

    def test_tracking_stability_monitor_flags_gaps_and_id_switches(self):
        monitor = radar.TrackingStabilityMonitor()
        monitor.update(
            radar.RadarFrame(
                frame_number=1,
                subframe_number=0,
                num_detected_obj=1,
                num_tlvs=1,
                points=[radar.RadarPoint(0.0, 2.0, 1.0, 0.0)],
                timestamp=1.0,
                presence=1,
            )
        )
        monitor.update(
            radar.RadarFrame(
                frame_number=2,
                subframe_number=0,
                num_detected_obj=1,
                num_tlvs=2,
                points=[radar.RadarPoint(0.0, 2.0, 1.0, 0.0)],
                timestamp=2.0,
                tracks=[radar.RadarTrack(track_id=1, x=0.0, y=2.0, z=1.0)],
                presence=1,
            )
        )
        monitor.update(
            radar.RadarFrame(
                frame_number=3,
                subframe_number=0,
                num_detected_obj=1,
                num_tlvs=2,
                points=[radar.RadarPoint(0.0, 2.0, 1.0, 0.0)],
                timestamp=3.0,
                tracks=[radar.RadarTrack(track_id=2, x=0.0, y=2.0, z=1.0)],
                presence=1,
            )
        )

        summary = monitor.to_dict()
        self.assertEqual(summary["frames_presence_without_tracks"], 1)
        self.assertEqual(summary["frames_points_without_tracks"], 1)
        self.assertEqual(summary["single_track_id_switches"], 1)
        self.assertEqual(summary["unique_track_ids"], [1, 2])


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

    def test_headless_frames_are_healthy_without_plot_updates(self):
        verdict, reason = radar.evaluate_health_verdict(
            mode="live",
            config_ok=True,
            last_command_error="",
            connection_error="",
            rx_bytes=4096,
            magic_hits=10,
            frames=8,
            last_parse_error="",
            data_opened_at=time.time() - 1.0,
            plots_updating=False,
        )
        self.assertEqual(verdict, radar.HEALTH_HEALTHY)
        self.assertIn("frames are parsing", reason.lower())


class DriverShutdownTests(unittest.TestCase):
    def test_open_serial_uses_quiet_non_flow_control_settings(self):
        if not radar._HAS_SERIAL:
            self.skipTest("pyserial is not installed")

        created = []
        old_serial = radar.serial.Serial

        class FakePySerial:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.port = kwargs.get("port")
                self.is_open = False
                self.exclusive = None
                self.events = []
                self._rts = None
                self._dtr = None
                created.append(self)

            @property
            def rts(self):
                return self._rts

            @rts.setter
            def rts(self, value):
                self._rts = bool(value)
                self.events.append(("rts", bool(value)))

            @property
            def dtr(self):
                return self._dtr

            @dtr.setter
            def dtr(self, value):
                self._dtr = bool(value)
                self.events.append(("dtr", bool(value)))

            def open(self):
                self.events.append(("open", self.port))
                self.is_open = True

        radar.serial.Serial = FakePySerial
        try:
            ser = radar.open_serial_with_retries(
                "/dev/ttyUSB0",
                115200,
                timeout=0.25,
                attempts=1,
            )
        finally:
            radar.serial.Serial = old_serial

        self.assertIs(ser, created[0])
        self.assertIsNone(ser.kwargs["port"])
        self.assertEqual(ser.kwargs["baudrate"], 115200)
        self.assertEqual(ser.kwargs["timeout"], 0.25)
        self.assertFalse(ser.kwargs["xonxoff"])
        self.assertFalse(ser.kwargs["rtscts"])
        self.assertFalse(ser.kwargs["dsrdtr"])
        self.assertEqual(ser.port, "/dev/ttyUSB0")
        self.assertTrue(ser.is_open)
        self.assertTrue(ser.exclusive)
        self.assertEqual(
            ser.events,
            [
                ("rts", False),
                ("dtr", False),
                ("open", "/dev/ttyUSB0"),
                ("rts", False),
                ("dtr", False),
            ],
        )

    def test_close_sends_sensor_stop_before_closing_ports(self):
        events = []
        cfg_ser = FakeSerialPort(name="cli", events=events)
        data_ser = FakeSerialPort(response=b"", name="data", events=events)
        driver = radar.IWR6843Driver()
        driver._config_serial = cfg_ser
        driver._data_serial = data_ser
        driver._running = True
        driver._connected = True
        radar._register_active_driver(driver)

        driver.close()

        self.assertIn(b"sensorStop\n", cfg_ser.writes)
        self.assertFalse(cfg_ser.is_open)
        self.assertFalse(data_ser.is_open)
        self.assertNotIn(driver, list(radar._ACTIVE_DRIVERS))
        self.assertEqual(
            events,
            [
                ("cli", "write", b"sensorStop\n"),
                ("cli", "close", b""),
                ("data", "close", b""),
            ],
        )

    def test_open_data_port_failure_stops_existing_sensor_before_close(self):
        if not radar._HAS_SERIAL:
            self.skipTest("pyserial is not installed")

        cfg_ser = FakeSerialPort()
        opened_ports = []
        old_open = radar.open_serial_with_retries
        old_settle = radar._CLI_POST_OPEN_SETTLE_S

        def fake_open_serial(port, baudrate, *, timeout, attempts=1, retry_delay_s=0.0):
            del baudrate, timeout, attempts, retry_delay_s
            opened_ports.append(port)
            if len(opened_ports) == 1:
                return cfg_ser
            raise radar.serial.SerialException("data port busy")

        radar.open_serial_with_retries = fake_open_serial
        radar._CLI_POST_OPEN_SETTLE_S = 0.0
        try:
            driver = radar.IWR6843Driver(
                config_port="cfg",
                data_port="data",
                config_path="iwr6843_people_tracking_20fps.cfg",
            )
            self.assertFalse(driver.open())
        finally:
            radar.open_serial_with_retries = old_open
            radar._CLI_POST_OPEN_SETTLE_S = old_settle

        self.assertEqual(opened_ports, ["cfg", "data"])
        self.assertIn(b"sensorStop\n", cfg_ser.writes)
        self.assertFalse(cfg_ser.is_open)


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

    def test_xds110_generic_cmsis_dap_pairing_from_location(self):
        infos = radar.discover_serial_ports(
            [
                SimpleNamespace(
                    device="/dev/ttyACM0",
                    description="XDS110 (03.00.00.02) Embed with CMSIS-DAP",
                    manufacturer="Texas Instruments",
                    product="",
                    interface="",
                    hwid="USB VID:PID=0451:BEF3 SER=R0081038 LOCATION=1-2.1:1.0",
                ),
                SimpleNamespace(
                    device="/dev/ttyACM1",
                    description="XDS110 (03.00.00.02) Embed with CMSIS-DAP",
                    manufacturer="Texas Instruments",
                    product="",
                    interface="",
                    hwid="USB VID:PID=0451:BEF3 SER=R0081038 LOCATION=1-2.1:1.3",
                ),
            ]
        )
        self.assertEqual(infos[0].role_hint, "config")
        self.assertEqual(infos[1].role_hint, "data")
        self.assertEqual(radar.suggest_serial_port_pairs(infos)[0], ("/dev/ttyACM0", "/dev/ttyACM1"))


if __name__ == "__main__":
    unittest.main()
