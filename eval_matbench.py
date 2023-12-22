import copy
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from crystal_gnn.config import ex
from crystal_gnn.datamodules import _datamodules
from crystal_gnn.models import _models


MB_TASKS = [
    "matbench_log_gvrh",
    "matbench_log_kvrh",
    "matbench_mp_e_form",
    "matbench_mp_gap",
    "matbench_mp_is_metal",
    "matbench_perovskites",
    "matbench_phonons",
]


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    project_name = _config["project_name"]
    exp_name = _config["exp_name"]
    log_dir = Path(_config["log_dir"], _config["source"])
    # set datamodule
    dm = _datamodules[_config["source"]](_config)
    # prepare data
    dm.prepare_data()
    # set model
    model = _models[_config["model_name"]](_config)
    print(model)
    # set checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/loss",
        mode="min",
        filename="best-{epoch}",
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    for fold in range(5):
        # set logger
        # set logger
        logger = WandbLogger(
            project=project_name,
            name=f"{exp_name}",
            version=f"{exp_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            if not _config["test_only"]
            else None,
            save_dir=log_dir,
            log_model="all",  # TODO: all or True?
            group=f"{_config['source']}-{_config['target']}-{_config['model_name']}",
        )
        # set trainer
        trainer = pl.Trainer(
            devices=_config["devices"],
            accelerator=_config["accelerator"],
            max_epochs=_config["max_epochs"],
            strategy="ddp_find_unused_parameters_true",
            deterministic=_config["deterministic"],
            callbacks=callbacks,
            logger=logger,
        )

        if not _config["test_only"]:
            dm.setup(stage="fit", fold=fold)
            trainer.fit(
                model,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(),
                ckpt_path=_config["resume_from"],
            )
            dm.setup(stage="test", fold=fold)
            trainer.test(
                model,
                dataloaders=dm.test_dataloader(),
                ckpt_path="best",
            )
        else:
            print(f"load model from {_config['load_path']}")
            dm.setup(stage="test", fold=fold)
            trainer.test(
                model,
                dataloaders=dm.test_dataloader(),
                ckpt_path=_config["load_path"],
            )

        # predict
        predictions = trainer.predict(
            model,
            dataloaders=dm.test_dataloader(),
            return_predictions=True,
        )
        predictions = torch.cat(predictions, dim=0)
        # record predictions
        task = dm.task

        task.record(fold, predictions)
        # temporary save json for matbench
        save_path = Path(
            _config["log_dir"],
            f"{_config['model_name']}_{task.dataset_name}",
            f"results_{_config['model_name']}_{task.dataset_name}.json.gz",
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dm.mb.to_file(save_path)
    # save json for matbench
    save_path = Path(
        _config["log_dir"],
        f"{_config['model_name']}_{task.dataset_name}",
        f"results_{_config['model_name']}_{task.dataset_name}.json.gz",
    )
    dm.mb.to_file(save_path)
    print(f"save matbench results to {save_path}")
