# this script is used to evaluate the performance of the models
# using Matbench datasets
import os
import copy

import torch
import pytorch_lightning as pl
from matbench.bench import MatbenchBenchmark

from crystal_gnn.config import ex
from crystal_gnn.models import _models
from crystal_gnn.datamodules.matbench_datamodule import MatbenchDataModule

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
    model_name = _config["model_name"]
    target = _config["target"]
    if target == "all":
        target_tasks = MB_TASKS
    else:
        if target in MB_TASKS:
            target_tasks = [target]
        else:
            raise ValueError(
                f"The target should be in {MB_TASKS}, Got {target} instead."
            )

    def get_matbench_task_by_name(name):
        for task in mb.tasks:
            if task.dataset_name == name:
                return task
        raise ValueError(f"Can't find {name} in MatbenchBenchmark.")

    for target_task in target_tasks:
        mb = MatbenchBenchmark(autoload=False)
        task = get_matbench_task_by_name(target_task)

        for fold in task.folds:
            # set datamodule
            dm = MatbenchDataModule(task, fold, _config)
            # set num_classes
            if task.metadata.task_type == "classification":
                _config["num_classes"] = 2
            else:
                _config["num_classes"] = 1
            # set mean and std for Normalizer
            if _config["num_classes"] == 1:
                _config["mean"] = dm.train_targets.mean()
                _config["std"] = dm.train_targets.std()
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
                name=f"{model_name}_{task.dataset_name}",
                version=f"fold_{fold}",
                default_hp_metric=False,
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
            # train
            trainer.fit(model, dm)
            # test
            trainer.test(model, dm, ckpt_path="best")
            # predict
            predictions = trainer.predict(model, dm, return_predictions=True)
            predictions = torch.cat(predictions, dim=0)
            # record predictions
            task.record(fold, predictions)
            # temporary save json for matbench
            save_path = os.path.join(
                _config["log_dir"],
                f"{model_name}_{task.dataset_name}",
                f"results_{model_name}_{task.dataset_name}.json.gz",
            )
            mb.to_file(save_path)
        # save json for matbench
        save_path = os.path.join(
            _config["log_dir"],
            f"{model_name}_{task.dataset_name}",
            f"results_{model_name}_{task.dataset_name}.json.gz",
        )
        mb.to_file(save_path)
        print(f"save matbench results to {save_path}")
