from lib.kits.hsmr_demo import *
from lib.platform.stage_profiler import StageProfiler

def main():
    # ⛩️ 0. Preparation.
    args = parse_args()
    outputs_root = Path(args.output_path)
    outputs_root.mkdir(parents=True, exist_ok=True)

    monitor = TimeMonitor()
    profiler = StageProfiler(enabled=args.profile, device=args.device)
    input_path = Path(args.input_path)
    profiler.set_metadata(
        input_path=str(input_path.resolve()),
        input_bytes=input_path.stat().st_size if input_path.is_file() else None,
        output_path=str(outputs_root.resolve()),
        device=args.device,
        detector_batch_size=args.det_bs,
        detector_max_image_size=args.det_mis,
        recovery_batch_size=args.rec_bs,
        max_instances=args.max_instances,
        include_skeleton=not args.ignore_skel,
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        gpu_name=(
            torch.cuda.get_device_name(torch.device(args.device))
            if args.device.startswith('cuda') and torch.cuda.is_available()
            else None
        ),
    )

    # ⛩️ 1. Preprocess.

    with monitor('Data Preprocessing'):
        with monitor('Load Inputs'):
            raw_imgs, inputs_meta = load_inputs(args, profiler=profiler)
            profiler.set_metadata(
                processed_frames=len(raw_imgs),
                processed_width=raw_imgs[0].shape[1],
                processed_height=raw_imgs[0].shape[0],
            )

        with monitor('Detector Initialization'):
            get_logger(brief=True).info('🧱 Building detector.')
            with profiler.measure('02_detector.init_model_weights_and_device', cuda=args.device.startswith('cuda')):
                detector = build_detector(
                        batch_size   = args.det_bs,
                        max_img_size = args.det_mis,
                        device       = args.device,
                        profiler     = profiler,
                    )

        with monitor('Detecting'):
            get_logger(brief=True).info(f'🖼️ Detecting...')
            detector_outputs = detector(raw_imgs)
            # The ViTDet-H detector is large. Release it before loading HSMR so
            # CPU-only machines do not need to keep both models in memory.
            with profiler.measure('03_detector.release_model_and_cuda_cache', cuda=args.device.startswith('cuda')):
                del detector
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()

        with monitor('Patching & Loading'):
            with profiler.measure('04_patches.cpu_filter_crop_and_resize'):
                patches, det_meta = imgs_det2patches(raw_imgs, *detector_outputs, args.max_instances)  # N * (256, 256, 3)
        if len(patches) == 0:
            get_logger(brief=True).error(f'🚫 No human instance detected. Please ensure the validity of your inputs!')
        get_logger(brief=True).info(f'🔍 Totally {len(patches)} human instances are detected.')
        profiler.set_metadata(detected_instances=len(patches))


    # ⛩️ 2. Human skeleton and mesh recovery.
    with monitor('Pipeline Initialization'):
        get_logger(brief=True).info(f'🧱 Building recovery pipeline.')
        with profiler.measure('05_hsmr.init_model_weights_and_device', cuda=args.device.startswith('cuda')):
            pipeline = build_inference_pipeline(
                model_root=args.model_root,
                device=args.device,
                profiler=profiler,
            )

    with monitor('Recovery'):
        get_logger(brief=True).info(f'🏃 Recovering with B={args.rec_bs}...')
        pd_params, pd_cam_t = [], []
        for bw in asb(total=len(patches), bs_scope=args.rec_bs, enable_tqdm=True):
            patches_i = patches[bw.sid:bw.eid]  # (N, 256, 256, 3)
            with profiler.measure('06_hsmr.cpu_normalize_and_transpose_patches'):
                patches_normalized_i = (patches_i - IMG_MEAN_255) / IMG_STD_255  # (N, 256, 256, 3)
                patches_normalized_i = np.ascontiguousarray(
                    patches_normalized_i.transpose(0, 3, 1, 2)
                )  # (N, 3, 256, 256)
            with torch.no_grad():
                outputs = pipeline(patches_normalized_i)
            with profiler.measure('06_hsmr.gpu_to_cpu_parameters', cuda=args.device.startswith('cuda')):
                pd_params.append({k: v.detach().cpu().clone() for k, v in outputs['pd_params'].items()})
                pd_cam_t.append(outputs['pd_cam_t'].detach().cpu().clone())

        with profiler.measure('06_hsmr.cpu_assemble_batch_results'):
            pd_params = assemble_dict(pd_params, expand_dim=False)  # [{k:[x]}, {k:[y]}] -> {k:[x, y]}
            pd_cam_t = torch.cat(pd_cam_t, dim=0)
            dump_data = {
                    'patch_cam_t' : pd_cam_t.numpy(),
                    **{k: v.numpy() for k, v in pd_params.items()},
                }

        get_logger(brief=True).info(f'🤌 Preparing meshes...')
        m_skin, m_skel = prepare_mesh(
            pipeline,
            pd_params,
            include_skeleton=not args.ignore_skel,
            profiler=profiler,
        )
        get_logger(brief=True).info(f'🏁 Done.')


    # ⛩️ 3. Postprocess.
    with monitor('Visualization'):
        if args.ignore_skel:
            m_skel = None
        results, full_cam_t = visualize_full_img(
            pd_cam_t,
            raw_imgs,
            det_meta,
            m_skin,
            m_skel,
            args.have_caption,
            profiler=profiler,
        )
        dump_data['full_cam_t'] = full_cam_t
        # Save rendering and dump results.
        if inputs_meta['type'] == 'video':
            seq_name = f'{pipeline.name}-' + inputs_meta['seq_name']
            save_video(
                results,
                outputs_root / f'{seq_name}.mp4',
                fps=inputs_meta.get('fps', 30),
                profiler=profiler,
            )
            # Dump data for each frame, here `i` refers to frames, `j` refers to image patches.
            with profiler.measure('09_output.cpu_build_prediction_records'):
                dump_results = []
                cur_patch_j = 0
                for i in range(len(raw_imgs)):
                    n_patch_cur_img = det_meta['n_patch_per_img'][i]
                    dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                    dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                    cur_patch_j += n_patch_cur_img
                    dump_results.append(dump_results_i)
            with profiler.measure('09_output.cpu_serialize_predictions'):
                np.save(outputs_root / f'{seq_name}.npy', dump_results)
        elif inputs_meta['type'] == 'imgs':
            img_names = [f'{pipeline.name}-{fn.name}' for fn in inputs_meta['img_fns']]
            # Dump data for each image separately, here `i` refers to images, `j` refers to image patches.
            cur_patch_j = 0
            for i, img_name in enumerate(tqdm(img_names, desc='Saving images')):
                n_patch_cur_img = det_meta['n_patch_per_img'][i]
                dump_results_i = {k: v[cur_patch_j:cur_patch_j+n_patch_cur_img] for k, v in dump_data.items()}
                dump_results_i['bbx_cs'] = det_meta['bbx_cs_per_img'][i]
                cur_patch_j += n_patch_cur_img
                with profiler.measure('09_output.cpu_save_images_and_predictions'):
                    save_img(results[i], outputs_root / f'{img_name}.jpg')
                    np.savez(outputs_root / f'{img_name}.npz', **dump_results_i)

        get_logger(brief=True).info(f'🎨 Rendering results are under {outputs_root}.')

    get_logger(brief=True).info(f'🎊 Everything is done!')
    profiler.finish()
    if args.profile:
        profile_output = Path(args.profile_output) if args.profile_output else outputs_root / 'profile'
        profile_paths = profiler.dump(profile_output)
        profiler.report()
        get_logger(brief=True).info(
            '⏱️ Detailed profiles: ' + ', '.join(str(path) for path in profile_paths)
        )
    monitor.report()


if __name__ == '__main__':
    main()
