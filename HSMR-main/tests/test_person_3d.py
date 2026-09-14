import math
import unittest

import numpy as np

from lib.platform.person_3d import (
    build_frame_3d_record,
    describe_root_orientation,
    rotation_matrix_to_quaternion_wxyz,
)


def rotation_y(angle):
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ])


class Person3DTest(unittest.TestCase):
    def test_quaternion_identity(self):
        quaternion = rotation_matrix_to_quaternion_wxyz(np.eye(3))
        np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_camera_relative_facing_angles(self):
        away = describe_root_orientation(np.eye(3), np.zeros(3))
        toward = describe_root_orientation(rotation_y(math.pi), np.zeros(3))
        image_right = describe_root_orientation(
            rotation_y(math.pi / 2.0),
            np.zeros(3),
        )

        self.assertAlmostEqual(abs(away['facing_camera_yaw_deg']), 180.0)
        self.assertEqual(away['coarse_facing'], 'away_from_camera')
        self.assertAlmostEqual(toward['facing_camera_yaw_deg'], 0.0)
        self.assertEqual(toward['coarse_facing'], 'toward_camera')
        self.assertAlmostEqual(image_right['facing_camera_yaw_deg'], 90.0)
        self.assertEqual(
            image_right['coarse_facing'],
            'toward_image_right',
        )

    def test_full_camera_coordinates_add_translation(self):
        body_data = {
            'joints_44': np.zeros((1, 44, 3)),
            'joints_24_anatomical': np.zeros((1, 24, 3)),
            'joints_24_custom': np.zeros((1, 24, 3)),
            'root_rotation': np.eye(3)[None],
        }
        body_data['joints_24_custom'][0, 0] = [0.1, -0.2, 0.3]
        full_translation = np.array([[1.0, 2.0, 3.0]])

        record = build_frame_3d_record(
            sample_id=1,
            source_sequence=2,
            captured_unix=3.0,
            image_shape=(480, 640, 3),
            bbx_cs=np.array([[320.0, 240.0, 200.0]]),
            patch_camera_translation=np.array([[0.0, 0.0, 2.0]]),
            full_camera_translation=full_translation,
            poses=np.zeros((1, 46)),
            body_data=body_data,
            backend='onnx',
            source='head',
            detection_scores=np.array([0.987654]),
            detector_bboxes_ltrb_px=np.array([[10.0, 20.0, 30.0, 40.0]]),
        )

        person = record['persons'][0]
        np.testing.assert_allclose(
            person['position']['pelvis_full_image_virtual_camera_m'],
            [1.1, 1.8, 3.3],
        )
        np.testing.assert_allclose(
            person['joints']['smpl_24_custom'][
                'full_image_virtual_camera_m'
            ][0],
            [1.1, 1.8, 3.3],
        )
        self.assertEqual(person['detection']['score'], 0.987654)
        self.assertEqual(
            person['detection']['bbox_left_top_right_bottom_px'],
            [10.0, 20.0, 30.0, 40.0],
        )


if __name__ == '__main__':
    unittest.main()
