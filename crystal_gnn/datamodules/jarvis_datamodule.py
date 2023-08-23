from typing import Any, Dict, Tuple, List

import os
import json
import math
import random

import numpy as np
import pandas as pd

from dgl.data import DGLDataset
from jarvis.db.figshare import data as jarvis_data

from crystal_gnn.datasets.jarvis_dataset import JarvisDataset
from crystal_gnn.datamodules.base_datamodule import BaseDataModule


class JarvisDataModule(BaseDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__(_config)

        self.database_name = _config["database_name"]
        self.target = _config["target"]
        self.data_dir = _config["data_dir"]
        self.classification_threshold = _config["classification_threshold"]
        self.split_seed = _config["split_seed"]
        self.train_ratio = _config["train_ratio"]
        self.val_ratio = _config["val_ratio"]
        self.test_ratio = _config["test_ratio"]
        self.keep_data_order = _config["keep_data_order"]

    def prepare_data(self) -> None:
        """Download data from JARVIS and split into train, val, test.
        Implemented in ALIGNN.

        It will save the dataset in the `data_dir/{target}` with the following names:
        - train-{database_name}-{target}.csv
        - val-{database_name}-{target}.csv
        - test-{database_name}-{target}.csv

        Args:
            database_name (str): name of the database in JARVIS
            target (str): target property in the database
            classification_threshold (Optional[float], optional): threshold for classification.
            split_seed (int, optional): seed for splitting data.
            train_ratio (float, optional): ratio of training data.
            val_ratio (float, optional): ratio of validation data.
            test_ratio (float, optional): ratio of test data.
            keep_data_order (bool, optional): whether to keep the order of data.
        """
        # check if jarvis data already exists
        path_data = os.path.join(
            self.data_dir, "database", f"{self.database_name}.json"
        )
        if os.path.exists(path_data):
            data = json.load(open(path_data))
            print(f"load data from {path_data}")
        else:
            # download data from JARVIS
            data = jarvis_data(self.database_name)
            os.makedirs(os.path.join(self.data_dir, "database"), exist_ok=True)
            json.dump(data, open(path_data, "w"))
            print(f"download data from JARVIS and save  to {path_data}")

        # check if the prepare_data has been done
        path_target = os.path.join(self.data_dir, self.target)
        path_train = os.path.join(
            path_target, f"train-{self.database_name}-{self.target}.csv"
        )
        path_val = os.path.join(
            path_target, f"val-{self.database_name}-{self.target}.csv"
        )
        path_test = os.path.join(
            path_target, f"test-{self.database_name}-{self.target}.csv"
        )
        path_info = os.path.join(
            path_target, f"{self.database_name}-{self.target}.json"
        )
        if (
            os.path.exists(path_train)
            and os.path.exists(path_val)
            and os.path.exists(path_test)
            and os.path.exists(path_info)
        ):
            print(f"load data from {path_target}")
            return

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

        id_train, id_val, id_test = get_id_train_val_test(
            total_size=len(dataset),
            split_seed=self.split_seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            keep_data_order=self.keep_data_order,
        )
        # split dataset
        if not os.path.exists(path_target):
            os.makedirs(path_target, exist_ok=True)

        dataset_train = pd.DataFrame([dataset[x] for x in id_train])[
            [id_key, "atoms", self.target]
        ]
        dataset_val = pd.DataFrame([dataset[x] for x in id_val])[
            [id_key, "atoms", self.target]
        ]
        dataset_test = pd.DataFrame([dataset[x] for x in id_test])[
            [id_key, "atoms", self.target]
        ]
        # make info dict
        info = (
            {
                "total": len(dataset),
                "train": len(dataset_train),
                "val": len(dataset_val),
                "test": len(dataset_test),
                "train_mean": dataset_train[self.target].mean(),
                "train_std": dataset_train[self.target].std(),
            },
        )
        # save dataset
        dataset_train.to_csv(path_train)
        dataset_val.to_csv(path_val)
        dataset_test.to_csv(path_test)
        json.dump(info, open(path_info, "w"))
        print(info)
        print(f"DONE: saved data to {path_target}")

    @property
    def dataset_cls(self) -> DGLDataset:
        return JarvisDataset

    @property
    def dataset_name(self) -> str:
        return "jarvis"


# get split indices
def get_id_train_val_test(
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
