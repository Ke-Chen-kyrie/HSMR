from lib.kits.hsmr_demo import *
from lib.kits.hsmr_demo import _img_det2patches
from lib.platform.stage_profiler import StageProfiler
from lib.platform.person_3d import (
    build_frame_3d_record,
    empty_frame_3d_record,
)

import csv
import json
import os
import resource
import threading
import time
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9501))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOXGLOVE_BRIDGE_URL = os.getenv(
    'FOXGLOVE_BRIDGE_URL',
    os.getenv('CAMERA_WS_URL', 'ws://192.168.217.100:8768'),
)
DEFAULT_CAMERA_TOPIC_HEAD = os.getenv(
    'CAMERA_TOPIC_HEAD',
    '/zj_humanoid/sensor/realsense_head/color/image_raw/compressed',
)
DEFAULT_CAMERA_TOPIC_UP = os.getenv(
    'CAMERA_TOPIC_UP',
    '/zj_humanoid/sensor/realsense_up/color/image_raw/compressed',
)
MEGABYTE = 1024.0 * 1024.0
JETSON_GPU_LOAD_PATHS = (
    Path('/sys/class/devfreq/17000000.gpu/load'),
    Path('/sys/devices/platform/bus@0/17000000.gpu/load'),
    Path('/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/load'),
    Path('/sys/devices/gpu.0/load'),
)
PERSON_3D_COLORS_RGB = (
    (255, 80, 80),
    (80, 255, 120),
    (80, 160, 255),
    (255, 210, 80),
    (220, 80, 255),
)


def safe_source_label(source):
    """Hide credentials when a stream URL is written to logs or metadata."""
    value = str(source)
    if '://' in value and '@' in value:
        scheme, remainder = value.split('://', 1)
        _, address = remainder.rsplit('@', 1)
        return f'{scheme}://***:***@{address}'
    return value


def annotate_person_3d(image_rgb, coordinates_3d):
    """Draw person IDs and camera-relative 3D summaries on every panel."""
    if not coordinates_3d['persons']:
        return image_rgb
    annotated = image_rgb.copy()
    raw_width = int(coordinates_3d['image']['width_px'])
    raw_height = int(coordinates_3d['image']['height_px'])
    panel_count = max(1, annotated.shape[1] // raw_width)

    for person in coordinates_3d['persons']:
        person_index = person['person_index']
        color = PERSON_3D_COLORS_RGB[
            person_index % len(PERSON_3D_COLORS_RGB)
        ]
        left, top, right, bottom = person[
            'crop_bbox_left_top_right_bottom_px'
        ]
        left = int(round(max(0.0, min(raw_width - 1.0, left))))
        right = int(round(max(0.0, min(raw_width - 1.0, right))))
        top = int(round(max(0.0, min(raw_height - 1.0, top))))
        bottom = int(round(max(0.0, min(raw_height - 1.0, bottom))))
        position = person['position']['pelvis_full_image_virtual_camera_m']
        orientation = person['orientation']
        score = person.get('detection', {}).get('score')
        score_text = f' score={score:.2f}' if score is not None else ''
        first_line = (
            f'P{person_index}{score_text} '
            f'xyz=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f})m'
        )
        second_line = (
            f'yaw={orientation["facing_camera_yaw_deg"]:+.1f}deg '
            f'{orientation["coarse_facing"]}'
        )
        label_y = max(42, top - 30)

        for panel_index in range(panel_count):
            x_offset = panel_index * raw_width
            cv2.rectangle(
                annotated,
                (left + x_offset, top),
                (right + x_offset, bottom),
                color,
                2,
            )
            cv2.putText(
                annotated,
                first_line,
                (left + x_offset, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                second_line,
                (left + x_offset, label_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )
    return annotated


def bytes_to_mb(value):
    return round(float(value) / MEGABYTE, 3)


def read_keyed_kilobytes(path):
    values = {}
    try:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                key, separator, remainder = line.partition(':')
                if not separator:
                    continue
                fields = remainder.strip().split()
                if fields:
                    try:
                        values[key] = float(fields[0])
                    except ValueError:
                        continue
    except OSError:
        pass
    return values


def read_jetson_gpu_load_percent():
    for path in JETSON_GPU_LOAD_PATHS:
        try:
            value = float(path.read_text(encoding='utf-8').strip())
        except (OSError, ValueError):
            continue
        # Jetson devfreq reports GPU load in the range [0, 1000].
        return max(0.0, min(100.0, value / 10.0))
    return None


def collect_memory_metrics(device):
    metrics = {}
    process_status = read_keyed_kilobytes('/proc/self/status')
    system_memory = read_keyed_kilobytes('/proc/meminfo')
    if 'VmRSS' in process_status:
        metrics['process_rss_mb'] = round(
            process_status['VmRSS'] / 1024.0,
            3,
        )
    # ru_maxrss is KiB on Linux.
    metrics['process_peak_rss_mb'] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        3,
    )
    if 'MemTotal' in system_memory:
        metrics['system_memory_total_mb'] = round(
            system_memory['MemTotal'] / 1024.0,
            3,
        )
    if 'MemAvailable' in system_memory:
        metrics['system_memory_available_mb'] = round(
            system_memory['MemAvailable'] / 1024.0,
            3,
        )

    if str(device).startswith('cuda') and torch.cuda.is_available():
        cuda_device = torch.device(device)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
            metrics.update({
                'cuda_memory_free_mb': bytes_to_mb(free_bytes),
                'cuda_memory_total_mb': bytes_to_mb(total_bytes),
                'cuda_memory_used_mb': bytes_to_mb(
                    total_bytes - free_bytes
                ),
            })
        except RuntimeError:
            pass
        metrics.update({
            # These four values cover PyTorch only. ONNX Runtime owns a
            # separate allocator and is reflected in CUDA free/used plus RSS.
            'torch_cuda_allocated_mb': bytes_to_mb(
                torch.cuda.memory_allocated(cuda_device)
            ),
            'torch_cuda_reserved_mb': bytes_to_mb(
                torch.cuda.memory_reserved(cuda_device)
            ),
            'torch_cuda_peak_allocated_mb': bytes_to_mb(
                torch.cuda.max_memory_allocated(cuda_device)
            ),
            'torch_cuda_peak_reserved_mb': bytes_to_mb(
                torch.cuda.max_memory_reserved(cuda_device)
            ),
        })
    return metrics


class RuntimePerformanceSampler:
    """Sample Jetson GPU load and collect per-frame memory statistics."""

    def __init__(self, device, interval_seconds=0.05):
        self.device = str(device)
        self.interval_seconds = float(interval_seconds)
        self.gpu_load_samples = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        self._sample_gpu_load()
        self.thread.start()
        return self

    def _sample_gpu_load(self):
        value = read_jetson_gpu_load_percent()
        if value is not None:
            self.gpu_load_samples.append(value)

    def _run(self):
        while not self.stopped.wait(self.interval_seconds):
            self._sample_gpu_load()

    def finish(self):
        self.stopped.set()
        self.thread.join(timeout=1.0)
        self._sample_gpu_load()
        metrics = collect_memory_metrics(self.device)
        if self.gpu_load_samples:
            metrics.update({
                'gpu_utilization_mean_percent': round(
                    sum(self.gpu_load_samples) / len(self.gpu_load_samples),
                    3,
                ),
                'gpu_utilization_peak_percent': round(
                    max(self.gpu_load_samples),
                    3,
                ),
                'gpu_utilization_samples': len(self.gpu_load_samples),
            })
        return metrics


class LatestFrameCapture:
    """Continuously decode a source and retain only its newest frame."""

    def __init__(
        self,
        source,
        width=None,
        height=None,
        camera_fps=None,
        camera_fourcc='MJPG',
        loop_file=False,
    ):
        self.source = source
        self.source_is_file = isinstance(source, str) and Path(source).is_file()
        self.loop_file = loop_file
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f'Cannot open camera/video source: {safe_source_label(source)}'
            )

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if isinstance(source, int) and camera_fourcc:
            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*camera_fourcc),
            )
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if camera_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, camera_fps)

        source_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.source_fps = source_fps if source_fps > 0 else 30.0
        self.frame = None
        self.sequence = -1
        self.captured_monotonic = None
        self.captured_unix = None
        self.error = None
        self.eof = False
        self.stopped = threading.Event()
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def properties(self):
        fourcc_value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = ''.join(
            chr((fourcc_value >> (8 * index)) & 0xFF)
            for index in range(4)
        )
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': float(self.cap.get(cv2.CAP_PROP_FPS)),
            'fourcc': fourcc,
        }

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        file_next_due = time.monotonic()
        while not self.stopped.is_set():
            if self.source_is_file:
                delay = file_next_due - time.monotonic()
                if delay > 0:
                    self.stopped.wait(delay)
                file_next_due += 1.0 / self.source_fps

            ok, frame = self.cap.read()
            if not ok:
                if self.source_is_file and self.loop_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    file_next_due = time.monotonic()
                    continue
                with self.condition:
                    self.eof = True
                    self.condition.notify_all()
                break

            with self.condition:
                self.frame = frame
                self.sequence += 1
                self.captured_monotonic = time.monotonic()
                self.captured_unix = time.time()
                self.condition.notify_all()

        self.cap.release()

    def snapshot(self, after_sequence=None, timeout=10.0):
        deadline = time.monotonic() + timeout
        with self.condition:
            while (
                not self.stopped.is_set()
                and not self.eof
                and (
                    self.frame is None
                    or (after_sequence is not None and self.sequence <= after_sequence)
                )
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

            if self.frame is None:
                return None
            return {
                'frame_bgr': self.frame.copy(),
                'sequence': self.sequence,
                'captured_monotonic': self.captured_monotonic,
                'captured_unix': self.captured_unix,
            }

    def stop(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=2.0)
        if self.thread.is_alive():
            self.cap.release()


def parse_source(value):
    return int(value) if value.lstrip('-').isdigit() else value


def parse_live_args():
    parser = argparse.ArgumentParser(
        description='Sample the latest camera frame periodically and run persistent HSMR inference.'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help=(
            'Camera index, RTSP/HTTP URL, video file, or Foxglove camera '
            'alias `head`/`up`.'
        ),
    )
    parser.add_argument(
        '--foxglove_url',
        type=str,
        default=DEFAULT_FOXGLOVE_BRIDGE_URL,
        help='Foxglove WebSocket Bridge URL used by --source head/up.',
    )
    parser.add_argument(
        '--foxglove_topic',
        type=str,
        default=None,
        help='Override the ROS2 image topic used by --source head/up.',
    )
    parser.add_argument('--interval', type=float, default=3.0, help='Sampling interval in seconds.')
    parser.add_argument('--max_frames', type=int, default=0, help='Stop after N inferred frames; 0 means run forever.')
    parser.add_argument('--output_path', type=str, default=str(PM.outputs/'webcam'), help='Output directory.')
    parser.add_argument('--model_root', type=str, default=DEFAULT_HSMR_ROOT)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        '--backend',
        choices=('pytorch', 'onnx'),
        default='pytorch',
        help='HSMR regression backend. The person detector and mesh renderer still use PyTorch.',
    )
    parser.add_argument(
        '--onnx_model',
        type=str,
        default=str(PROJECT_ROOT / 'deploy/onnx/artifacts/hsmr_full_fp16.onnx'),
        help='Full or core HSMR ONNX graph used when --backend onnx.',
    )
    parser.add_argument(
        '--onnx_provider',
        type=str,
        default='auto',
        help=(
            'ONNX Runtime provider: auto, cpu, cuda, tensorrt, or an exact '
            'provider name such as CUDAExecutionProvider.'
        ),
    )
    parser.add_argument('--det_mis', type=int, default=512)
    parser.add_argument(
        '--no_detector_amp',
        action='store_false',
        dest='detector_amp',
        help='Disable FP16 autocast for the live detector.',
    )
    parser.set_defaults(detector_amp=True)
    parser.add_argument('--max_instances', type=int, default=1)
    parser.add_argument('--mesh_bs', type=int, default=1)
    parser.add_argument('--ignore_skel', action='store_true', help='Skip skeleton generation and rendering.')
    parser.add_argument('--width', type=int, default=None, help='Requested camera width.')
    parser.add_argument('--height', type=int, default=None, help='Requested camera height.')
    parser.add_argument('--camera_fps', type=float, default=None, help='Requested camera FPS.')
    parser.add_argument('--camera_fourcc', type=str, default='MJPG', help='Requested USB camera FOURCC.')
    parser.add_argument('--loop_file', action='store_true', help='Loop when --source is a video file.')
    parser.add_argument('--no_history', action='store_true', help='Only update latest.jpg instead of saving every result.')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be greater than zero.')
    if args.max_frames < 0:
        parser.error('--max_frames must be zero or positive.')
    if len(args.camera_fourcc) != 4:
        parser.error('--camera_fourcc must contain exactly four characters.')
    return args


def atomic_save_rgb_jpg(image_rgb, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.parent / f'.{output_path.name}.tmp.jpg'
    ok = cv2.imwrite(
        str(temporary_path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )
    if not ok:
        raise RuntimeError(f'Failed to save image: {temporary_path}')
    os.replace(temporary_path, output_path)


def atomic_save_json(data, output_path):
    output_path = Path(output_path)
    temporary_path = output_path.parent / f'.{output_path.name}.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, output_path)


def append_latency_csv(output_path, row):
    output_path = Path(output_path)
    fieldnames = [
        'sample_id',
        'source_sequence',
        'captured_unix',
        'capture_age_ms',
        'persons',
        'end_to_end_ms',
        'detector_ms',
        'hsmr_ms',
        'mesh_ms',
        'render_ms',
        'save_ms',
        'gpu_utilization_mean_percent',
        'gpu_utilization_peak_percent',
        'cuda_memory_used_mb',
        'cuda_memory_free_mb',
        'cuda_memory_total_mb',
        'torch_cuda_allocated_mb',
        'torch_cuda_reserved_mb',
        'torch_cuda_peak_allocated_mb',
        'torch_cuda_peak_reserved_mb',
        'process_rss_mb',
        'process_peak_rss_mb',
        'system_memory_available_mb',
        'system_memory_total_mb',
        'deadline_missed',
    ]
    if output_path.exists():
        with open(output_path, 'r', newline='', encoding='utf-8') as file:
            existing_reader = csv.DictReader(file)
            existing_fieldnames = existing_reader.fieldnames or []
            if existing_fieldnames != fieldnames:
                existing_rows = list(existing_reader)
                temporary_path = (
                    output_path.parent / f'.{output_path.name}.migrate'
                )
                with open(
                    temporary_path,
                    'w',
                    newline='',
                    encoding='utf-8',
                ) as migrated_file:
                    migrated_writer = csv.DictWriter(
                        migrated_file,
                        fieldnames=fieldnames,
                    )
                    migrated_writer.writeheader()
                    for existing_row in existing_rows:
                        migrated_writer.writerow({
                            key: existing_row.get(key)
                            for key in fieldnames
                        })
                os.replace(temporary_path, output_path)

    new_file = not output_path.exists()
    with open(output_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})
        file.flush()


def stage_total(profile_data, prefix):
    return sum(
        stage['total_seconds']
        for name, stage in profile_data['stages'].items()
        if name.startswith(prefix)
    )


def infer_latest_frame(
    frame_bgr,
    detector,
    pipeline,
    args,
    sample_id,
    source_sequence,
    captured_monotonic,
    captured_unix,
):
    profiler = StageProfiler(enabled=True, device=args.device)
    detector.profiler = profiler
    pipeline.stage_profiler = profiler
    profiler.set_metadata(
        sample_id=sample_id,
        source_sequence=source_sequence,
        captured_unix=captured_unix,
        device=args.device,
        backend=args.backend,
        include_skeleton=not args.ignore_skel,
    )

    capture_age_ms = (time.monotonic() - captured_monotonic) * 1000.0
    with profiler.measure('01_live.cpu_bgr_to_rgb'):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detector_outputs = detector([frame_rgb])
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    with profiler.measure('04_patches.cpu_filter_crop_and_resize'):
        patches, bbx_cs, detection_meta = _img_det2patches(
            frame_rgb,
            detector_outputs[0][0],#只保留pred_class == 0的人物
            detector_outputs[1][0],
            args.max_instances,#最多保留max_instances的人物
            return_detection_meta=True,
        )

    persons = len(patches)
    if persons == 0:
        with profiler.measure('08_render.cpu_no_detection_annotation'):
            rendered = frame_rgb.copy()
            cv2.putText(
                rendered,
                'No person detected',
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 80, 80),
                2,
                cv2.LINE_AA,
            )
        with profiler.measure('09_output.cpu_build_3d_record'):
            coordinates_3d = empty_frame_3d_record(
                sample_id=sample_id,
                source_sequence=source_sequence,
                captured_unix=captured_unix,
                image_shape=frame_rgb.shape,
                backend=args.backend,
                source=safe_source_label(args.source),
            )
        profiler.finish()
        return (
            rendered,
            profiler.summary(),
            persons,
            capture_age_ms,
            coordinates_3d,
        )

    patches = patches.astype(np.float32)
    with profiler.measure('06_hsmr.cpu_normalize_and_transpose_patches'):
        patches_normalized = (patches - IMG_MEAN_255) / IMG_STD_255
        patches_normalized = np.ascontiguousarray(
            patches_normalized.transpose(0, 3, 1, 2)
        )

    with torch.no_grad():
        outputs = pipeline(patches_normalized)
    with profiler.measure(
        '06_hsmr.gpu_to_cpu_parameters',
        cuda=args.device.startswith('cuda'),
    ):
        pd_params = {
            key: value.detach().cpu().clone()
            for key, value in outputs['pd_params'].items()
        }
        pd_cam_t = outputs['pd_cam_t'].detach().cpu().clone()

    m_skin, m_skel, body_data = prepare_mesh(
        pipeline,
        pd_params,
        include_skeleton=not args.ignore_skel,
        batch_size=args.mesh_bs,
        profiler=profiler,
        return_body_data=True,
    )
    det_meta = {
        'n_patch_per_img': [persons],
        'bbx_cs_per_img': [bbx_cs],
        'bbx_cs': np.asarray(bbx_cs),
    }
    rendered, raw_cam_t = visualize_full_img(
        pd_cam_t,
        [frame_rgb],
        det_meta,
        m_skin,
        m_skel,
        profiler=profiler,
    )
    with profiler.measure('09_output.cpu_build_3d_record'):
        coordinates_3d = build_frame_3d_record(
            sample_id=sample_id,
            source_sequence=source_sequence,
            captured_unix=captured_unix,
            image_shape=frame_rgb.shape,
            bbx_cs=bbx_cs,
            patch_camera_translation=pd_cam_t,
            full_camera_translation=raw_cam_t,
            poses=pd_params['poses'],
            body_data=body_data,
            backend=args.backend,
            source=safe_source_label(args.source),
            detection_scores=detection_meta['scores'],
            detector_bboxes_ltrb_px=detection_meta[
                'detector_bboxes_ltrb_px'
            ],
        )
    with profiler.measure('08_render.cpu_3d_annotation'):
        rendered_frame = annotate_person_3d(rendered[0], coordinates_3d)
    profiler.finish()
    return (
        rendered_frame,
        profiler.summary(),
        persons,
        capture_age_ms,
        coordinates_3d,
    )


def main():
    args = parse_live_args()
    source = parse_source(args.source)
    outputs_root = Path(args.output_path)
    history_root = outputs_root / 'frames'
    outputs_root.mkdir(parents=True, exist_ok=True)
    if not args.no_history:
        history_root.mkdir(parents=True, exist_ok=True)

    get_logger(brief=True).info(
        f'🧱 Loading persistent detector and HSMR {args.backend} backend once.'
    )
    startup_started = time.perf_counter()
    detector_started = time.perf_counter()
    detector = build_detector(
        batch_size=1,
        max_img_size=args.det_mis,
        device=args.device,
        use_amp=args.detector_amp,
    )
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(torch.device(args.device))
    detector_init_seconds = time.perf_counter() - detector_started

    pipeline_started = time.perf_counter()
    if args.backend == 'onnx':
        from deploy.onnx.runtime import HSMRONNXRuntimePipeline

        pipeline = HSMRONNXRuntimePipeline(
            model_path=args.onnx_model,
            model_root=args.model_root,
            device=args.device,
            provider=args.onnx_provider,
        )
        get_logger(brief=True).info(
            f'ONNX Runtime graph={pipeline.graph_scope}, '
            f'provider={pipeline.provider}, model={pipeline.model_path}'
        )
    else:
        pipeline = build_inference_pipeline(
            model_root=args.model_root,
            device=args.device,
        )
    if args.device.startswith('cuda'):
        torch.cuda.synchronize(torch.device(args.device))
    hsmr_init_seconds = time.perf_counter() - pipeline_started
    startup_data = {
        'detector_init_seconds': detector_init_seconds,
        'hsmr_init_seconds': hsmr_init_seconds,
        'total_startup_seconds': time.perf_counter() - startup_started,
        'device': args.device,
        'backend': args.backend,
        'gpu_name': (
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith('cuda') and torch.cuda.is_available()
            else None
        ),
        'source': safe_source_label(source),
        'interval_seconds': args.interval,
        'include_skeleton': not args.ignore_skel,
        'detector_amp': args.detector_amp,
        'save_history': not args.no_history,
        'history_path': str(history_root) if not args.no_history else None,
        'memory_after_model_load': collect_memory_metrics(args.device),
        'performance_notes': [
            (
                'Jetson uses unified CPU/GPU memory. CUDA used/free values '
                'are device-wide rather than process-exclusive, and include '
                'all CUDA contexts and allocators.'
            ),
            (
                'torch_cuda_* values cover PyTorch only; ONNX Runtime uses '
                'a separate allocator and is reflected by CUDA used/free and '
                'process RSS.'
            ),
        ],
    }
    if args.backend == 'onnx':
        startup_data.update({
            'onnx_model': str(pipeline.model_path),
            'onnx_graph_scope': pipeline.graph_scope,
            'onnx_provider': pipeline.provider,
            'onnx_providers': pipeline.providers,
        })
    foxglove_alias = (
        source.lower()
        if isinstance(source, str) and source.lower() in ('head', 'up')
        else None
    )
    if foxglove_alias is not None:
        from lib.platform.foxglove_camera import FoxgloveLatestFrameCapture

        default_topic = (
            DEFAULT_CAMERA_TOPIC_HEAD
            if foxglove_alias == 'head'
            else DEFAULT_CAMERA_TOPIC_UP
        )
        foxglove_topic = args.foxglove_topic or default_topic
        capture = FoxgloveLatestFrameCapture(
            url=args.foxglove_url,
            topic=foxglove_topic,
        )
        startup_data['source'] = f'foxglove:{foxglove_alias}'
        get_logger(brief=True).info(
            f'📡 Foxglove camera: url={safe_source_label(args.foxglove_url)}, '
            f'topic={foxglove_topic}'
        )
    else:
        capture = LatestFrameCapture(
            source=source,
            width=args.width,
            height=args.height,
            camera_fps=args.camera_fps,
            camera_fourcc=args.camera_fourcc,
            loop_file=args.loop_file,
        )
    capture_properties = capture.properties()
    if 'url' in capture_properties:
        capture_properties['url'] = safe_source_label(
            capture_properties['url']
        )
    startup_data['capture'] = capture_properties
    atomic_save_json(startup_data, outputs_root / 'startup.json')
    capture.start()
    first = capture.snapshot(timeout=15.0)
    if first is None:
        capture.stop()
        capture_error = (
            f' Last capture error: {capture.error}'
            if capture.error
            else ''
        )
        raise RuntimeError(
            f'No frame received from source: {safe_source_label(source)}.'
            f'{capture_error}'
        )
    capture_properties = capture.properties()
    if 'url' in capture_properties:
        capture_properties['url'] = safe_source_label(
            capture_properties['url']
        )
    startup_data['capture'] = capture_properties
    atomic_save_json(startup_data, outputs_root / 'startup.json')

    get_logger(brief=True).info(
        f'📷 Live inference started: source={safe_source_label(source)}, '
        f'interval={args.interval:.3f}s. '
        'Press Ctrl+C to stop.'
    )
    sample_id = 0
    last_sequence = -1
    next_due = time.monotonic()
    try:
        while args.max_frames == 0 or sample_id < args.max_frames:
            wait_seconds = next_due - time.monotonic()
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            snapshot = capture.snapshot(after_sequence=last_sequence, timeout=max(2.0, args.interval))
            if snapshot is None:
                if capture.eof:
                    get_logger(brief=True).info('Video source reached EOF.')
                    break
                get_logger(brief=True).warning('No new camera frame arrived before timeout.')
                next_due += args.interval
                continue

            sample_id += 1
            last_sequence = snapshot['sequence']
            frame_started = time.perf_counter()
            performance_sampler = RuntimePerformanceSampler(
                args.device
            ).start()
            (
                rendered,
                profile_data,
                persons,
                capture_age_ms,
                coordinates_3d,
            ) = infer_latest_frame(
                frame_bgr=snapshot['frame_bgr'],
                detector=detector,
                pipeline=pipeline,
                args=args,
                sample_id=sample_id,
                source_sequence=last_sequence,
                captured_monotonic=snapshot['captured_monotonic'],
                captured_unix=snapshot['captured_unix'],
            )

            save_started = time.perf_counter()
            atomic_save_rgb_jpg(rendered, outputs_root / 'latest.jpg')
            atomic_save_json(coordinates_3d, outputs_root / 'latest_3d.json')
            history_coordinates_path = None
            if not args.no_history:
                timestamp = datetime.fromtimestamp(
                    snapshot['captured_unix']
                ).strftime('%Y%m%d-%H%M%S-%f')[:-3]
                history_stem = f'{sample_id:06d}-{timestamp}'
                atomic_save_rgb_jpg(
                    rendered,
                    history_root / f'{history_stem}.jpg',
                )
                history_coordinates_path = (
                    history_root / f'{history_stem}.3d.json'
                )
                atomic_save_json(
                    coordinates_3d,
                    history_coordinates_path,
                )
            save_seconds = time.perf_counter() - save_started
            end_to_end_seconds = time.perf_counter() - frame_started
            performance_data = performance_sampler.finish()
            profile_data['metadata']['performance'] = performance_data

            row = {
                'sample_id': sample_id,
                'source_sequence': last_sequence,
                'captured_unix': snapshot['captured_unix'],
                'capture_age_ms': capture_age_ms,
                'persons': persons,
                'end_to_end_ms': end_to_end_seconds * 1000.0,
                'detector_ms': stage_total(profile_data, '03_detector.') * 1000.0,
                'hsmr_ms': stage_total(profile_data, '06_hsmr.') * 1000.0,
                'mesh_ms': stage_total(profile_data, '07_mesh.') * 1000.0,
                'render_ms': stage_total(profile_data, '08_render.') * 1000.0,
                'save_ms': max(0.0, save_seconds) * 1000.0,
                **performance_data,
                'deadline_missed': end_to_end_seconds > args.interval,
            }
            latest_data = {
                **row,
                'performance': performance_data,
                'profile': profile_data,
                'coordinates_3d_latest_path': str(
                    outputs_root / 'latest_3d.json'
                ),
                'coordinates_3d_history_path': (
                    str(history_coordinates_path)
                    if history_coordinates_path is not None
                    else None
                ),
                'persons_3d_summary': [
                    {
                        'person_index': person['person_index'],
                        'crop_bbox_left_top_right_bottom_px': person[
                            'crop_bbox_left_top_right_bottom_px'
                        ],
                        'detection': person.get('detection'),
                        'position': person['position'],
                        'orientation': {
                            'facing_camera_yaw_deg': person[
                                'orientation'
                            ]['facing_camera_yaw_deg'],
                            'forward_elevation_deg': person[
                                'orientation'
                            ]['forward_elevation_deg'],
                            'coarse_facing': person[
                                'orientation'
                            ]['coarse_facing'],
                            'body_forward_unit_model_camera': person[
                                'orientation'
                            ]['body_forward_unit_model_camera'],
                        },
                    }
                    for person in coordinates_3d['persons']
                ],
            }
            atomic_save_json(latest_data, outputs_root / 'latest.json')
            with open(outputs_root / 'latency.jsonl', 'a', encoding='utf-8') as file:
                file.write(json.dumps(latest_data, ensure_ascii=False) + '\n')
                file.flush()
            append_latency_csv(outputs_root / 'latency.csv', row)

            persons_3d_log = ', '.join(
                (
                    f'p{person["person_index"]} '
                    f'pelvis={person["position"]["pelvis_full_image_virtual_camera_m"]}m '
                    f'yaw={person["orientation"]["facing_camera_yaw_deg"]}deg '
                    f'facing={person["orientation"]["coarse_facing"]}'
                )
                for person in coordinates_3d['persons']
            ) or 'none'
            get_logger(brief=True).info(
                f'✅ sample={sample_id} source_frame={last_sequence} persons={persons} '
                f'latency={end_to_end_seconds:.3f}s capture_age={capture_age_ms:.1f}ms '
                f'gpu_peak={performance_data.get("gpu_utilization_peak_percent")}%, '
                f'cuda_used={performance_data.get("cuda_memory_used_mb")}MiB, '
                f'rss={performance_data.get("process_rss_mb")}MiB, '
                f'deadline_missed={row["deadline_missed"]}, '
                f'3d={persons_3d_log}'
            )

            next_due += args.interval
            now = time.monotonic()
            if next_due <= now:
                skipped_slots = int((now - next_due) // args.interval) + 1
                next_due += skipped_slots * args.interval
                get_logger(brief=True).warning(
                    f'Inference exceeded the schedule; skipped {skipped_slots} stale slot(s).'
                )
    except KeyboardInterrupt:
        get_logger(brief=True).info('Stopping on Ctrl+C.')
    finally:
        capture.stop()
        detector.profiler = None
        pipeline.stage_profiler = None

    get_logger(brief=True).info(
        f'🎊 Live inference stopped after {sample_id} samples. Results: {outputs_root}'
    )


if __name__ == '__main__':
    main()
