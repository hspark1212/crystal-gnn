from typing import List
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch


class MatbenchDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        source: str,
        target: str,
        split: str,
        **kwargs,
    ):
        """Generate MatbenchDataset.

        Args:
            data_dir (str): directory to save the dataset
            source (str): source of the database
            target (str): target property in the database
            split (str): split of the dataset, one of train, val, test
        """
        super().__init__()
        # get fold from kwargs for Matbench
        fold = kwargs.get("fold", None)
        if fold is None:
            raise ValueError("fold must be provided for MatbenchDataset")

        # read graph data
        path_data = Path(data_dir, source, target, f"{split}-{target}-fold{fold}.pt")
        if not path_data.exists():
            raise FileNotFoundError(f"{path_data} does not exist")
        self.graphs = torch.load(path_data)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Data:
        return self.graphs[index]

    @staticmethod
    def collate_fn(batch: List[Data]) -> Batch:
        return Batch.from_data_list(batch)
