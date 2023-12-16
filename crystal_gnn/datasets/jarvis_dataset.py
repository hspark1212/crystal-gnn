from typing import Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class JarvisDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        source: str,
        target: str,
        split: str,
        **kwargs,
    ) -> None:
        """Generate JarvisDataset.

        Args:
            data_dir (str): directory to save the dataset
            source (str): source of the database
            target (str): target property in the database
            split (str): split of the dataset, one of train, val, test
        """
        super().__init__()
        # get database name from kwargs for Jarvis
        database_name = kwargs.get("database_name", None)
        if database_name is None:
            raise ValueError("database_name must be provided for JarvisDataset")

        # read graph data
        path_data = Path(
            data_dir, source, target, f"{split}-{database_name}-{target}.pt"
        )
        if not path_data.exists():
            raise FileNotFoundError(f"{path_data} does not exist")
        self.graphs = torch.load(path_data)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = dict()
        graph = self.graphs[index]
        target = graph.y
        data.update({"graph": graph, "target": target})

        return data

    @staticmethod
    def collate_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """batch collate function"""
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # get batch graph
        dict_batch["graph"] = Batch(dict_batch["graph"])
        # get batch target
        dict_batch["target"] = torch.tensor(dict_batch["target"])
        return dict_batch
