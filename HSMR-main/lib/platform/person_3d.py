"""Serialize HSMR/SKEL per-person positions, joints, and root orientation.

The values produced here deliberately keep HSMR's virtual camera frame
separate from a calibrated RGB-D/robot frame.  A monocular HSMR camera
translation is useful for rendering and relative geometry, but it is not a
RealSense depth measurement and is not a ROS ``base_link`` transform.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping

import numpy as np


OPENPOSE25_JOINT_NAMES = [
    'nose',
    'neck',
    'right_shoulder',
    'right_elbow',
    'right_wrist',
    'left_shoulder',
    'left_elbow',
    'left_wrist',
    'mid_hip',
    'right_hip',
    'right_knee',
    'right_ankle',
    'left_hip',
    'left_knee',
    'left_ankle',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
]

# These are the 19 rows produced by the project's SMPL_to_J19 regressor.
EXTRA19_JOINT_NAMES = [
    'right_ankle_smpl',
    'right_knee_smpl',
    'right_hip_smpl',
    'left_hip_smpl',
    'left_knee_smpl',
    'left_ankle_smpl',
    'right_wrist_smpl',
    'right_elbow_smpl',
    'right_shoulder_smpl',
    'left_shoulder_smpl',
    'left_elbow_smpl',
    'left_wrist_smpl',
    'neck_lsp',
    'top_of_head_lsp',
    'pelvis_mpii',
    'thorax_mpii',
    'spine_h36m',
    'jaw_h36m',
    'head_h36m',
]

HSMR44_JOINT_NAMES = OPENPOSE25_JOINT_NAMES + EXTRA19_JOINT_NAMES

SKEL24_ANATOMICAL_JOINT_NAMES = [
    'pelvis',
    'femur_r',
    'tibia_r',
    'talus_r',
    'calcn_r',
    'toes_r',
    'femur_l',
    'tibia_l',
    'talus_l',
    'calcn_l',
    'toes_l',
    'lumbar_body',
    'thorax',
    'head',
    'scapula_r',
    'humerus_r',
    'ulna_r',
    'radius_r',
    'hand_r',
    'scapula_l',
    'humerus_l',
    'ulna_l',
    'radius_l',
    'hand_l',
]

SMPL24_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hand',
    'right_hand',
]

POSE_PARAMETER_NAMES = [
    'pelvis_tilt',
    'pelvis_list',
    'pelvis_rotation',
]


def _as_numpy(value: Any, dtype=np.float64) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _rounded_list(value: Any, digits: int = 6) -> list:
    return np.round(_as_numpy(value), decimals=digits).tolist()


def rotation_matrix_to_quaternion_wxyz(matrix: Any) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized wxyz quaternion."""
    m = _as_numpy(matrix)
    if m.shape != (3, 3):
        raise ValueError(f'Expected a 3x3 rotation matrix, got {m.shape}.')

    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ])
    else:
        diagonal_index = int(np.argmax(np.diag(m)))
        if diagonal_index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ])
        elif diagonal_index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ])

    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def facing_label(facing_camera_yaw_deg: float) -> str:
    """Return a coarse label for a precisely defined camera-relative yaw."""
    angle = float(facing_camera_yaw_deg)
    if abs(angle) <= 45.0:
        return 'toward_camera'
    if abs(angle) >= 135.0:
        return 'away_from_camera'
    return 'toward_image_right' if angle > 0.0 else 'toward_image_left'


def describe_root_orientation(
    root_rotation: Any,
    root_pose_parameters: Any,
) -> Dict[str, Any]:
    """Describe the root transform in unambiguous camera-relative terms.

    HSMR's canonical body points forward along +Z, upward along +Y, and
    toward the anatomical left along +X.
    ``root_rotation`` maps those canonical axes into the HSMR model/virtual
    camera axes.  The virtual camera uses +X image-right, +Y image-down and
    +Z away from the camera.
    """
    rotation = _as_numpy(root_rotation)
    pose_parameters = _as_numpy(root_pose_parameters).reshape(3)
    body_left = rotation @ np.array([1.0, 0.0, 0.0])
    body_right = -body_left
    body_up = rotation @ np.array([0.0, 1.0, 0.0])
    body_forward = rotation @ np.array([0.0, 0.0, 1.0])

    horizontal_norm = math.hypot(body_forward[0], body_forward[2])
    # 0 degrees means the person faces the camera (-Z); positive turns the
    # forward vector toward image-right (+X), negative toward image-left.
    camera_yaw = math.degrees(
        math.atan2(body_forward[0], -body_forward[2])
    )
    forward_elevation = math.degrees(
        math.atan2(-body_forward[1], horizontal_norm)
    )
    quaternion = rotation_matrix_to_quaternion_wxyz(rotation)

    return {
        'skel_root_pose_parameter_names': POSE_PARAMETER_NAMES,
        'skel_root_pose_radians': _rounded_list(pose_parameters),
        'skel_root_pose_degrees': _rounded_list(
            np.degrees(pose_parameters),
            digits=3,
        ),
        'rotation_matrix_body_canonical_to_model_camera': _rounded_list(
            rotation,
        ),
        'quaternion_body_canonical_to_model_camera_wxyz': _rounded_list(
            quaternion,
        ),
        'body_forward_unit_model_camera': _rounded_list(body_forward),
        'body_up_unit_model_camera': _rounded_list(body_up),
        'body_left_unit_model_camera': _rounded_list(body_left),
        'body_right_unit_model_camera': _rounded_list(body_right),
        'facing_camera_yaw_deg': round(camera_yaw, 3),
        'forward_elevation_deg': round(forward_elevation, 3),
        'coarse_facing': facing_label(camera_yaw),
    }


def _joint_coordinates(
    model_coordinates: Any,
    full_camera_translation: np.ndarray,
) -> Dict[str, list]:
    model_coordinates = _as_numpy(model_coordinates)
    return {
        'model_origin_relative_m': _rounded_list(model_coordinates),
        'full_image_virtual_camera_m': _rounded_list(
            model_coordinates + full_camera_translation[None, :],
        ),
    }


def _joint_set_metadata() -> Dict[str, Mapping[str, Any]]:
    return {
        'hsmr_44': {
            'count': 44,
            'names': HSMR44_JOINT_NAMES,
            'description': (
                'OpenPose BODY_25 ordering followed by the 19 rows from '
                'SMPL_to_J19.pkl.'
            ),
        },
        'skel_24_anatomical': {
            'count': 24,
            'names': SKEL24_ANATOMICAL_JOINT_NAMES,
            'description': 'Native biomechanical SKEL joint centers.',
        },
        'smpl_24_custom': {
            'count': 24,
            'names': SMPL24_JOINT_NAMES,
            'description': (
                'HSMR custom/SMPL-style joint centers used by the wrapper.'
            ),
        },
    }


def build_frame_3d_record(
    *,
    sample_id: int,
    source_sequence: int,
    captured_unix: float,
    image_shape: Iterable[int],
    bbx_cs: Any,
    patch_camera_translation: Any,
    full_camera_translation: Any,
    poses: Any,
    body_data: Mapping[str, Any],
    backend: str,
    source: str,
    detection_scores: Any = None,
    detector_bboxes_ltrb_px: Any = None,
    virtual_focal_length_px: float = 5000.0,
) -> Dict[str, Any]:
    """Build the JSON-ready 3D record for one camera frame."""
    image_height, image_width = list(image_shape)[:2]
    boxes = _as_numpy(bbx_cs)
    patch_cam = _as_numpy(patch_camera_translation)
    full_cam = _as_numpy(full_camera_translation)
    pose_array = _as_numpy(poses)
    joints_44 = _as_numpy(body_data['joints_44'])
    joints_24_anatomical = _as_numpy(body_data['joints_24_anatomical'])
    joints_24_custom = _as_numpy(body_data['joints_24_custom'])
    root_rotation = _as_numpy(body_data['root_rotation'])
    scores = (
        _as_numpy(detection_scores).reshape(-1)
        if detection_scores is not None
        else None
    )
    detector_boxes = (
        _as_numpy(detector_bboxes_ltrb_px)
        if detector_bboxes_ltrb_px is not None
        else None
    )

    people_count = len(boxes)
    expected_shapes = {
        'patch_camera_translation': (people_count, 3),
        'full_camera_translation': (people_count, 3),
        'poses': (people_count, 46),
        'joints_44': (people_count, 44, 3),
        'joints_24_anatomical': (people_count, 24, 3),
        'joints_24_custom': (people_count, 24, 3),
        'root_rotation': (people_count, 3, 3),
    }
    actual_values = {
        'patch_camera_translation': patch_cam,
        'full_camera_translation': full_cam,
        'poses': pose_array,
        'joints_44': joints_44,
        'joints_24_anatomical': joints_24_anatomical,
        'joints_24_custom': joints_24_custom,
        'root_rotation': root_rotation,
    }
    for name, expected_shape in expected_shapes.items():
        if actual_values[name].shape != expected_shape:
            raise ValueError(
                f'{name} has shape {actual_values[name].shape}; '
                f'expected {expected_shape}.'
            )
    if scores is not None and scores.shape != (people_count,):
        raise ValueError(
            f'detection_scores has shape {scores.shape}; '
            f'expected {(people_count,)}.'
        )
    if detector_boxes is not None and detector_boxes.shape != (people_count, 4):
        raise ValueError(
            f'detector_bboxes_ltrb_px has shape {detector_boxes.shape}; '
            f'expected {(people_count, 4)}.'
        )

    persons = []
    for person_index in range(people_count):
        cx, cy, scale = boxes[person_index]
        full_translation = full_cam[person_index]
        pelvis_model_coordinates = joints_24_custom[person_index, 0]
        pelvis_full_camera = pelvis_model_coordinates + full_translation
        distance = float(np.linalg.norm(pelvis_full_camera))

        person = {
            'person_index': person_index,
            'crop_bbox_center_scale_xy_px': _rounded_list(
                [cx, cy, scale],
                digits=3,
            ),
            'crop_bbox_left_top_right_bottom_px': _rounded_list(
                [
                    cx - scale / 2.0,
                    cy - scale / 2.0,
                    cx + scale / 2.0,
                    cy + scale / 2.0,
                ],
                digits=3,
            ),
            'position': {
                'model_origin_crop_virtual_camera_m': _rounded_list(
                    patch_cam[person_index],
                ),
                'model_origin_full_image_virtual_camera_m': _rounded_list(
                    full_translation,
                ),
                'pelvis_model_origin_relative_m': _rounded_list(
                    pelvis_model_coordinates,
                ),
                'pelvis_full_image_virtual_camera_m': _rounded_list(
                    pelvis_full_camera,
                ),
                'pelvis_optical_axis_depth_m': round(
                    float(pelvis_full_camera[2]),
                    6,
                ),
                'pelvis_distance_from_virtual_camera_m': round(distance, 6),
            },
            'orientation': describe_root_orientation(
                root_rotation[person_index],
                pose_array[person_index, :3],
            ),
            'joints': {
                'hsmr_44': _joint_coordinates(
                    joints_44[person_index],
                    full_translation,
                ),
                'skel_24_anatomical': _joint_coordinates(
                    joints_24_anatomical[person_index],
                    full_translation,
                ),
                'smpl_24_custom': _joint_coordinates(
                    joints_24_custom[person_index],
                    full_translation,
                ),
            },
        }
        if scores is not None or detector_boxes is not None:
            person['detection'] = {}
            if scores is not None:
                person['detection']['score'] = round(
                    float(scores[person_index]),
                    6,
                )
            if detector_boxes is not None:
                person['detection'][
                    'bbox_left_top_right_bottom_px'
                ] = _rounded_list(
                    detector_boxes[person_index],
                    digits=3,
                )
        persons.append(person)

    return {
        'schema': 'hsmr_webcam_3d_v1',
        'sample_id': int(sample_id),
        'source_sequence': int(source_sequence),
        'captured_unix': float(captured_unix),
        'source': str(source),
        'backend': str(backend),
        'persons_count': people_count,
        'image': {
            'width_px': int(image_width),
            'height_px': int(image_height),
            'virtual_intrinsics_px': {
                'fx': float(virtual_focal_length_px),
                'fy': float(virtual_focal_length_px),
                'cx': float(image_width / 2.0),
                'cy': float(image_height / 2.0),
            },
        },
        'coordinate_frames': {
            'hsmr_model_translation_zero': {
                'origin': 'HSMR/SKEL model origin with trans=0',
                'axes': '+X image-right, +Y image-down, +Z away from camera',
                'note': (
                    'The predicted global root orientation is already '
                    'applied to these joint coordinates. The model origin '
                    'is not the pelvis joint; use the explicitly exported '
                    'pelvis coordinates when a pelvis position is required.'
                ),
            },
            'full_image_virtual_camera': {
                'origin': 'virtual pinhole camera optical center',
                'axes': '+X image-right, +Y image-down, +Z away from camera',
                'transform': (
                    'p_full_virtual = p_model_origin_relative + '
                    'model_origin_full_image_virtual_camera_m'
                ),
                'warning': (
                    'Monocular estimate using a fixed virtual focal length; '
                    'not RealSense depth, ROS camera optical frame, map, '
                    'odom, or robot base_link.'
                ),
            },
            'body_canonical_for_orientation': {
                'axes': '+X body-left, +Y body-up, +Z body-forward',
                'note': (
                    'The exported rotation matrix maps this canonical frame '
                    'to the HSMR model/virtual-camera axes.'
                ),
            },
        },
        'units_note': (
            'Geometry uses SKEL model metres. Absolute monocular translation '
            'and depth remain scale/focal-length dependent estimates.'
        ),
        'joint_sets': _joint_set_metadata(),
        'persons': persons,
    }


def empty_frame_3d_record(
    *,
    sample_id: int,
    source_sequence: int,
    captured_unix: float,
    image_shape: Iterable[int],
    backend: str,
    source: str,
) -> Dict[str, Any]:
    """Create the same schema for a frame in which no person was detected."""
    image_height, image_width = list(image_shape)[:2]
    return {
        'schema': 'hsmr_webcam_3d_v1',
        'sample_id': int(sample_id),
        'source_sequence': int(source_sequence),
        'captured_unix': float(captured_unix),
        'source': str(source),
        'backend': str(backend),
        'persons_count': 0,
        'image': {
            'width_px': int(image_width),
            'height_px': int(image_height),
        },
        'joint_sets': _joint_set_metadata(),
        'persons': [],
        'note': 'No person passed the detector threshold in this frame.',
    }
