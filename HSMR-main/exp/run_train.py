from lib.kits.basic import *

from pytorch_lightning.loggers import TensorBoardLogger

from lib.utils.media import *
from lib.platform import entrypoint_with_args


@entrypoint_with_args(exp='hsmr/train')
def main(cfg:DictConfig):
    seed = cfg.get('seed', None)
    if seed is not None:
        pl.seed_everything(seed)

    get_logger().info('Setting data module...')
    data_module = instantiate(cfg.data, _recursive_=False)

    get_logger().info('Setting pipeline...')
    pipeline = instantiate(cfg.pipeline, _recursive_=False)
    pipeline.set_data_adaption(data_module.name)

    if cfg.ckpt_path is not None:
        get_logger().info(f'Loading checkpoint from: {cfg.ckpt_path}')
        pipeline.load_state_dict(torch.load(cfg.ckpt_path)['state_dict'])

    logger = TensorBoardLogger(
            # save_dir = PM.outputs / 'tb_logs',
            save_dir = Path(cfg.output_dir) / 'tb_logs',
            name     = cfg.exp_name,
            version  = '',
        )

    callbacks = []
    get_logger().info('Setting callbacks...')
    for cb_name in cfg.callbacks:
        cb_cfg = cfg.callbacks[cb_name]
        get_logger().info(f'- Init callback: {cb_name}')
        callbacks.append(instantiate(cb_cfg, _recursive_=False))

    get_logger().info('Setting pl trainer...')
    trainer = pl.Trainer(
            accelerator = 'gpu',
            logger      = logger if logger is not None else False,
            callbacks   = callbacks,
            **cfg.pl_trainer,
        )

    trainer.fit(
        model      = pipeline,
        datamodule = data_module,
        ckpt_path  = 'last',
    )


if __name__ == '__main__':
    main()