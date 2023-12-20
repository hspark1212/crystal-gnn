from typing import Dict, Any
from pathlib import Path
import json

from ase import Atoms

import torch
from torch_geometric.data import Dataset

from matbench import MatbenchBenchmark

from crystal_gnn.datasets.matbench_dataset import MatbenchDataset
from crystal_gnn.datamodules.base_datamodule import BaseDataModule


class MatbenchDataModule(BaseDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)

        self.source = _config["source"]
        self.target = _config["target"]  # task
        self.data_dir = _config["data_dir"]
        self.split_seed = _config["split_seed"]
        self.train_ratio = _config["train_ratio"]
        self.val_ratio = _config["val_ratio"]
        self.test_ratio = _config["test_ratio"]
        self.database_name = None  # this is not used for Matbench

        # load matbench data
        self.mb = MatbenchBenchmark(autoload=False, subset=[self.target])
        self.mb.load()
        self.task = list(self.mb.tasks)[0]

    def prepare_data(self) -> None:
        """Download data from MATBENCH and split into train, val, test.

        It will save the torch_geometric graph data for train, val, test
        in the `{data_dir}/{source}/{target}` with the following names:
        - train-{target}_fold{fold}.pt
        - val-{target}_fold{fold}.pt
        - test-{target}_fold{fold}.pt
        - {target}_fold{fold}.json (info files)
        """
        # make path_target if not exists
        path_target = Path(self.data_dir, self.source, self.target)
        if not path_target.exists():
            path_target.mkdir(parents=True, exist_ok=True)

        for fold in self.task.folds:
            # check if the prepared data already exists
            path_train = Path(path_target, f"train-{self.target}-fold{fold}.pt")
            path_val = Path(path_target, f"val-{self.target}-fold{fold}.pt")
            path_test = Path(path_target, f"test-{self.target}-fold{fold}.pt")
            path_info = Path(path_target, f"{self.target}-fold{fold}.pt")
            if (
                path_train.exists()
                and path_val.exists()
                and path_test.exists()
                and path_info.exists()
            ):
                print(f"load graph data from {path_target} for fold {fold}")
                continue

            inputs, outputs = self.task.get_train_and_val_data(fold)
            # split train and val data
            randperm = torch.randperm(len(inputs)).tolist()
            num_train = int(len(inputs) * self.train_ratio)
            train_inputs = inputs[randperm[:num_train]]
            train_outputs = outputs[randperm[:num_train]]
            val_inputs = inputs[randperm[num_train:]]
            val_outputs = outputs[randperm[num_train:]]
            # make train data
            train_names, train_structures = zip(*train_inputs.items())
            train_targets = train_outputs.values
            # make val data
            val_names, val_structures = zip(*val_inputs.items())
            val_targets = val_outputs.values
            # get test data
            test_inputs, test_outputs = self.task.get_test_data(
                fold, include_target=True
            )
            # make test data
            test_names, test_structures = zip(*test_inputs.items())
            test_targets = test_outputs.values

            # save graph data for train, val, test
            for split, names, structures, targets in zip(
                ["train", "val", "test"],
                [train_names, val_names, test_names],
                [train_structures, val_structures, test_structures],
                [train_targets, val_targets, test_targets],
            ):
                # convert Structure to ase Atoms
                atoms_list = [self.convert_to_ase_atoms(s) for s in structures]
                if split == "train":
                    train_mean = targets.mean()
                    train_std = targets.std()

                # make graph data
                graph_data = self._make_graph_data(
                    atoms_list,
                    target=targets,
                    name=names,
                    train_mean=train_mean,
                    train_std=train_std,
                )
                # save graph
                path_split = Path(path_target, f"{split}-{self.target}-fold{fold}.pt")
                torch.save(graph_data, path_split)

            # save info
            info = {
                "total": len(train_names) + len(val_names) + len(test_names),
                "train": len(train_names),
                "val": len(val_names),
                "test": len(test_names),
                "train_mean": train_mean,
                "train_std": train_std,
            }
            json.dump(info, open(path_info, "w"))
            print(info)

    @property
    def dataset_cls(self) -> Dataset:
        return MatbenchDataset

    @property
    def dataset_name(self) -> str:
        return "matbench"

    @classmethod
    def convert_to_ase_atoms(cls, structure) -> Atoms:
        return Atoms(
            numbers=structure.atomic_numbers,
            positions=structure.cart_coords,
            cell=structure.lattice.matrix,
            pbc=True,
        )
