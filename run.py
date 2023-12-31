import copy

import pytorch_lightning as pl

from crystal_gnn.config import ex
from crystal_gnn.datamodules import _datamodules
from crystal_gnn.models import _models


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    exp_name = _config["exp_name"]
    # set datamodule
    dm = _datamodules[_config["source"]](_config)
    # prepare data
    dm.prepare_data()
    # set model
    model = _models[_config["model_name"]](_config)
    print(model)
    # set checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/loss",
        mode="min",
        filename="best-{epoch}",
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]
    # set logger
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f"{exp_name}",
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
        trainer.fit(model, dm, ckpt_path=_config["resume_from"])
        trainer.test(model, datamodule=dm, ckpt_path="best")
    else:
        model.load_from_checkpoint(_config["load_path"])
        print(f"load model from {_config['load_path']}")
        trainer.test(model, datamodule=dm)
