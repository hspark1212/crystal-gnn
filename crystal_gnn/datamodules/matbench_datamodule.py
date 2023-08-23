from typing import Dict, List, Any

from pymatgen.core import Structure

from matbench.task import MatbenchTask

import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader


from crystal_gnn.datasets.matbench_dataset import MatbenchDataset
from crystal_gnn.datamodules.base_datamodule import BaseDataModule


class MatbenchDataModule(BaseDataModule):
    def __init__(self, task: MatbenchTask, fold: int, _config: Dict[str, Any]) -> None:
        super().__init__(_config)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.task = task
        self.fold = fold
        self.train_ratio = _config["train_ratio"]
        self.val_ratio = _config["val_ratio"]
        self.test_ratio = _config["test_ratio"]
        # load data
        self.task.load()
        # get train and val data
        inputs, outputs = self.task.get_train_and_val_data(self.fold)
        # split train and val data
        randperm = torch.randperm(len(inputs)).tolist()
        num_train = int(len(inputs) * self.train_ratio)
        train_inputs = inputs[randperm[:num_train]]
        train_outputs = outputs[randperm[:num_train]]
        val_inputs = inputs[randperm[num_train:]]
        val_outputs = outputs[randperm[num_train:]]
        # make train data
        self.train_names = list(train_inputs.keys())
        self.train_structures = list(train_inputs.values)
        self.train_targets = torch.Tensor(list(train_outputs.values))
        # make val data
        self.val_names = list(val_inputs.keys())
        self.val_structures = list(val_inputs.values)
        self.val_targets = torch.Tensor(list(val_outputs.values))
        # get test data
        test_inputs, test_outputs = self.task.get_test_data(
            self.fold, include_target=True
        )
        # make test data
        self.test_names = list(test_inputs.keys())
        self.test_structures = list(test_inputs.values)
        self.test_targets = torch.Tensor(list(test_outputs.values))

    @property
    def dataset_cls(self) -> DGLDataset:
        return MatbenchDataset

    @property
    def dataset_name(self) -> str:
        return "matbench"

    def _set_dataset(
        self,
        names: List[str],
        structures: List[Structure],
        targets: List[Any],
        split: str,
    ) -> DGLDataset:
        print(f"Start setting {split} dataset...")
        return MatbenchDataset(
            names=names,
            structures=structures,
            targets=targets,
            split=split,
            compute_line_graph=self.compute_line_graph,
            neighbor_strategy=self.neighbor_strategy,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            use_canonize=self.use_canonize,
        )

    def setup(self, stage: str = None) -> None:
        # train
        if stage == "fit" or stage is None:
            self.train_dataset = self._set_dataset(
                names=self.train_names,
                structures=self.train_structures,
                targets=self.train_targets,
                split="train",
            )
            self.val_dataset = self._set_dataset(
                names=self.val_names,
                structures=self.val_structures,
                targets=self.val_targets,
                split="val",
            )
        # test
        elif stage == "test":
            self.test_dataset = self._set_dataset(
                names=self.test_names,
                structures=self.test_structures,
                targets=self.test_targets,
                split="test",
            )
        # predict
        elif stage == "predict":
            self.test_dataset = self._set_dataset(
                names=self.test_names,
                structures=self.test_structures,
                targets=self.test_targets,
                split="test",
            )
            self.use_ddp = False  # pylint: disable=attribute-defined-outside-init
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def _set_dataloader(
        self,
        dataset: DGLDataset,
        shuffle: bool,
    ) -> GraphDataLoader:
        return GraphDataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=dataset.collate_fn,
            use_ddp=self.use_ddp,
            drop_last=False,
        )

    def train_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> GraphDataLoader:
        return self._set_dataloader(self.test_dataset, shuffle=False)
