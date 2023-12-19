from typing import Any, Dict, Tuple, List
from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd

from ase import Atoms

import torch
from torch_geometric.data import Dataset

from jarvis.db.figshare import data as jarvis_data

from crystal_gnn.datamodules.base_datamodule import BaseDataModule
from crystal_gnn.datasets.jarvis_dataset import JarvisDataset


class JarvisDataModule(BaseDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)

        self.data_dir = _config["data_dir"]
        self.source = _config["source"]
        self.database_name = _config["database_name"]
        self.target = _config["target"]
        self.classification_threshold = _config["classification_threshold"]
        self.split_seed = _config["split_seed"]
        self.train_ratio = _config["train_ratio"]
        self.val_ratio = _config["val_ratio"]
        self.test_ratio = _config["test_ratio"]

    def prepare_data(self) -> None:
        """Download data from JARVIS and split into train, val, test.

        (1) It will save the dataset in the `{data_dir}/{database}` with the following names:
        - {database_name}.json (whloe dataset for {database_name} from JARVIS)

        (2) It will save the torch_geometric graph data for train, val, test
        in the `{data_dir}/{source}/{target}` with the following names:
        - train-{database_name}-{target}.pt
        - val-{database_name}-{target}.pt
        - test-{database_name}-{target}.pt
        - {database_name}-{target}.json (info files)
        """
        # check if jarvis data already exists
        path_database = Path(
            self.data_dir, self.source, "database", f"{self.database_name}.json"
        )
        if path_database.exists():
            data = json.load(open(path_database))
            print(f"load data from {path_database}")
        else:
            # download data from JARVIS
            data = jarvis_data(self.database_name)
            Path(self.data_dir, self.source, "database").mkdir(
                parents=True, exist_ok=True
            )
            json.dump(data, open(path_database, "w"))
            print(f"download data from JARVIS and save to {path_database}")

        # check if the prepare_data has been done
        path_target = Path(self.data_dir, self.source, self.target)
        path_train = Path(path_target, f"train-{self.database_name}-{self.target}.pt")
        path_val = Path(path_target, f"val-{self.database_name}-{self.target}.pt")
        path_test = Path(path_target, f"test-{self.database_name}-{self.target}.pt")
        path_info = Path(path_target, f"{self.database_name}-{self.target}.json")
        if (
            path_train.exists()
            and path_val.exists()
            and path_test.exists()
            and path_info.exists()
        ):
            print(f"load graph data from {path_target}")
            return
        # make path_target if not exists
        if not path_target.exists():
            Path(path_target).mkdir(parents=True, exist_ok=True)

        # make dataset
        keys = list(data[0].keys())
        print(f"{self.database_name} has keys: {keys}")
        if "jid" in keys:
            id_key = "jid"
        elif "id" in keys:
            id_key = "id"
        else:
            raise ValueError("No id key found in the database")
        # remove data with missing target values
        dataset = [
            i for i in data if i[self.target] != "na" and not math.isnan(i[self.target])
        ]
        # convert classification target to 0/1 if classification_threshold
        if self.classification_threshold is not None:
            for i in dataset:
                if i[self.target] <= self.classification_threshold:
                    i[self.target] = 0
                elif i[self.target] > self.classification_threshold:
                    i[self.target] = 1

        id_train, id_val, id_test = self.get_id_train_val_test(
            total_size=len(dataset),
            split_seed=self.split_seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            keep_data_order=False,  # It should be False
        )

        # save graph data for train, val, test
        for split, id_split in zip(
            ["train", "val", "test"], [id_train, id_val, id_test]
        ):
            df = pd.DataFrame([dataset[x] for x in id_split])
            # convert jarvis atoms to ase Atoms
            atoms_list = df["atoms"].apply(self.convert_to_ase_atoms).values
            # calculate mean and std for train data
            if split == "train":
                train_mean = df[self.target].mean()
                train_std = df[self.target].std()
            # make graph data
            graphs = self._make_graph_data(
                atoms_list,
                target=df[self.target].values,
                name=df[id_key].values,
                train_mean=train_mean,
                train_std=train_std,
            )
            # save graph
            path_split = Path(
                path_target, f"{split}-{self.database_name}-{self.target}.pt"
            )
            torch.save(graphs, path_split)
            print(f"DONE: saved data to {path_split}")

        # save info
        info = (
            {
                "total": len(id_train) + len(id_val) + len(id_test),
                "train": len(id_train),
                "val": len(id_val),
                "test": len(id_test),
                "train_mean": train_mean,
                "train_std": train_std,
            },
        )
        json.dump(info, open(path_info, "w"))
        print(info)
        print(f"DONE: saved data to {path_target}")

    @property
    def dataset_cls(self) -> Dataset:
        return JarvisDataset

    @property
    def dataset_name(self) -> str:
        return "jarvis"

    @classmethod
    def convert_to_ase_atoms(cls, jarvis_atoms) -> Atoms:
        return Atoms(
            symbols=jarvis_atoms["elements"],
            positions=jarvis_atoms["coords"],
            cell=jarvis_atoms["lattice_mat"],
            pbc=True,
        )

    # get split indices
    def get_id_train_val_test(
        self,
        total_size: int,
        split_seed: int,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        keep_data_order: bool,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Split a dataset into training, validation, and test sets,
        maintaining the order of data if specified.
        To make consistent splits, we use the same function in ALIGNN.
        """
        # Validate input
        assert (
            0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1
        ), "Ratios must be between 0 and 1"
        assert train_ratio + val_ratio + test_ratio == 1.0, "Sum of ratios must be 1"

        # Set default number of elements for each subset
        n_train = int(train_ratio * total_size)
        n_test = int(test_ratio * total_size)
        n_val = total_size - n_train - n_test

        # Generate IDs
        ids = list(np.arange(total_size))
        if not keep_data_order:
            random.seed(split_seed)
            random.shuffle(ids)

        # Split into subsets
        id_train = ids[:n_train]
        id_val = ids[n_train : n_train + n_val]
        id_test = ids[n_train + n_val :]

        return id_train, id_val, id_test
