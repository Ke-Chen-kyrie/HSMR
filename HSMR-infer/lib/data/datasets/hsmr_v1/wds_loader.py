from lib.kits.basic import *

import webdataset as wds

from .utils import *
from .stream_pipelines import *

# This line is to fix the problem of "OSError: image file is truncated" when loading images.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_tars_as_wds(
    cfg        : DictConfig,
    urls       : Union[str, List[str]],
    resampled  : bool = False,
    epoch_size : int  = None,
    cache_dir  : str  = None,
    train      : bool = True,
):
    urls = expand_urls(urls)  # to list of URL strings

    dataset : wds.WebDataset = wds.WebDataset(
            urls,
            nodesplitter = wds.split_by_node,
            shardshuffle = True,
            resampled    = resampled,
            cache_dir    = cache_dir,
        )
    if train:
        dataset = dataset.shuffle(100)

    # A lot of processes to initialize the dataset. Check the pipeline generator function for more details.
    # The order of the pipeline is important, since some of the process are dependent on some previous ones.
    dataset = apply_corrupt_filter(dataset)
    dataset = dataset.decode('rgb8').rename(jpg='jpg;jpeg;png')
    dataset = apply_multi_ppl_splitter(dataset)
    dataset = apply_keys_adapter(dataset)  #* This adapter is only in HSMR's design, not in the baseline.
    dataset = apply_bad_pgt_params_nan_suppressor(dataset)
    dataset = apply_bad_pgt_params_kp2d_err_suppressor(dataset, cfg.get('suppress_pgt_params_kp2d_err_thresh', 0.0))
    dataset = apply_bad_pgt_params_pve_max_suppressor(dataset, cfg.get('suppress_pgt_params_pve_max_thresh', 0.0))
    dataset = apply_bad_kp_suppressor(dataset, cfg.get('suppress_kp_conf_thresh', 0.0))
    dataset = apply_bad_betas_suppressor(dataset, cfg.get('suppress_betas_thresh', 0.0))
    # dataset = apply_bad_pose_suppressor(dataset, cfg.get('suppress_pose_thresh', 0.0))  # Not used in baseline, so not implemented.
    dataset = apply_params_synchronizer(dataset, cfg.get('poses_betas_simultaneous', False))
    # dataset = apply_no_pose_filter(dataset, cfg.get('no_pose_filter', False))  # Not used in baseline, so not implemented.
    dataset = apply_insuff_kp_filter(dataset, cfg.get('filter_insufficient_kp_cnt', 4), cfg.get('suppress_insufficient_kp_thresh', 0.0))
    dataset = apply_bbox_size_filter(dataset, cfg.get('filter_bbox_size_thresh', None))
    dataset = apply_reproj_err_filter(dataset, cfg.get('filter_reproj_err_thresh', 0.0))
    dataset = apply_invalid_betas_regularizer(dataset, cfg.get('regularize_invalid_betas', False))

    # Final preprocess / format of the data. (Consider to extract the augmentation process.)
    dataset = apply_example_formatter(dataset, cfg)

    if epoch_size is not None:
        dataset = dataset.with_epoch(epoch_size)
    return dataset