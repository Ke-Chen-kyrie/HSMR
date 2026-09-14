# Training

Before start, please make sure you can launch the demo and have additionally prepared 🟣 items listed [here](./SETUP.md#overview).

The default experiment setting for training our ViT-H backbone is listed at [`configs/exp/hsmr/train.yaml`](../configs/exp/hsmr/train.yaml).

In normal cases, you might want to adjust `data.cfg.train.dataloader.batch_size` and `pl_trainer.devices` to fit your hardware.

```shell
python exp/run_train.py  # The experiment name will be `{exp_topic}-{exp_tag}`.
```

You will see the config menu after launching the training script. In order to modify the settings, you have two choices:

1. Modify the config file directly.
2. Use the CLI arguments to override the config items.
3. Create a new (nested) config file and specify it in the CLI arguments.

Please refer to the [Hydra documentation](https://hydra.cc/docs/intro) for more details about CLI arguments.


## Configurations System

HSMR's framwork have some special design for the configuration system based on Hydra. Refer to [`configs/README.md`](../configs/README.md) for more details.

Here are the notes for each configuration files.

- `_hub_`: meta source for most common resources (like datasets, body models, etc.). Basically you don't want to make chance here.
- `callback`
    - `ckpt`: simple configurations to define how to save the checkpoints.
    - `skelify-spin`: different configurations for the `skelify-spin` module, i.e., the iterative pseudo-G.T. refinement module.
- `data`: define the composition and the setting of the datasets for training.
- `exp`: experiment entry configurations. Most configurations for training process can be found here.
- `pipeline`: define the pipeline-specific setting.
- `policy`: some shared constants.
- `base.yaml`: ignore this, it's a functional but content-less file.