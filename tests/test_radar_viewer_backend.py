import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import radar
import radar_viewer


class RadarViewerBackendTests(unittest.TestCase):
    def test_default_cfg_path_points_at_existing_cfg(self):
        cfg_path = radar_viewer._default_cfg_path()
        self.assertTrue(cfg_path)
        self.assertTrue(Path(cfg_path).exists())

    def test_backend_bootstrap_has_a_clear_result(self):
        self.assertTrue(radar_viewer._HAS_QT or radar_viewer._QT_IMPORT_ERROR is not None)

    def test_fixed_room_box_expands_scene_boxes_without_reframing_each_frame(self):
        box = radar.RadarBox3D(
            x_min=-1.0,
            x_max=0.5,
            y_min=1.5,
            y_max=4.0,
            z_min=0.1,
            z_max=2.3,
            kind="boundary",
        )
        frame = radar.RadarFrame(
            frame_number=1,
            subframe_number=0,
            num_detected_obj=0,
            num_tlvs=0,
            points=[],
            timestamp=0.0,
            scene=radar.RadarSceneMetadata(boundary_boxes=[box]),
        )
        room = radar_viewer._fixed_room_box_for_scene(frame)
        self.assertLessEqual(room.x_min, -1.0)
        self.assertGreaterEqual(room.x_max, 0.5)
        self.assertLessEqual(room.y_min, 0.0)
        self.assertGreaterEqual(room.y_max, 4.0)
        self.assertLessEqual(room.z_min, 0.0)
        self.assertGreaterEqual(room.z_max, 2.3)
        self.assertGreaterEqual(room.width, radar_viewer.DEFAULT_ROOM_WIDTH_M)
        self.assertGreaterEqual(room.depth, radar_viewer.DEFAULT_ROOM_DEPTH_M)
        self.assertGreaterEqual(room.height, radar_viewer.DEFAULT_ROOM_HEIGHT_M)

    def test_box_mesh_geometry_builds_closed_prism(self):
        vertices, faces = radar_viewer._box_mesh_geometry(
            radar.RadarBox3D(
                x_min=-0.4,
                x_max=0.4,
                y_min=1.0,
                y_max=2.0,
                z_min=0.0,
                z_max=1.8,
            )
        )
        self.assertEqual(vertices.shape, (8, 3))
        self.assertEqual(faces.shape, (12, 3))

    def test_room_guides_produce_floor_and_wall_segments(self):
        segments = radar_viewer._room_guide_segments(radar_viewer._default_room_box(), spacing=2.0)
        self.assertGreater(len(segments), 0)
        self.assertEqual(segments.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
