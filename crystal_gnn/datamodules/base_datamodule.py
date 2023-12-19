from typing import Dict, Any, List, Iterable
from tqdm import tqdm

from ase import Atoms
from ase.neighborlist import neighbor_list

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # configs for prepare_data
        self.cutoff = _config["cutoff"]

        # configs for dataloader
        self.batch_size = _config["batch_size"]
        self.num_workers = _config["num_workers"]
        self.pin_memory = _config["pin_memory"]

    @property
    def dataset_cls(self) -> Dataset:
        raise NotImplementedError

    @property
    def dataset_name(self) -> str:
        raise NotImplementedError

    def _set_dataset(self, split: str, **kwargs) -> Dataset:
        print(f"Start setting {split} dataset...")
        return self.dataset_cls(
            data_dir=self.data_dir,
            source=self.source,
            target=self.target,
            split=split,
            database_name=self.database_name,
            **kwargs,  # fold, database_name
        )

    def setup(self, stage: str = None, **kwargs) -> None:
        """
        Args:
            stage (str): fit, test, predict
        kwargs:
            - fold (int): fold number for Matbench
            - database_name (str): database name for JARVIS
        """
        # train
        if stage == "fit":
            self.train_dataset = self._set_dataset(split="train", **kwargs)
            self.val_dataset = self._set_dataset(split="val", **kwargs)
        # test
        elif stage == "test":
            self.test_dataset = self._set_dataset(split="test", **kwargs)
        # predict
        elif stage == "predict":
            self.test_dataset = self._set_dataset(split="test", **kwargs)
        else:
            self.train_dataset = self._set_dataset(split="train", **kwargs)
            self.val_dataset = self._set_dataset(split="val", **kwargs)
            self.test_dataset = self._set_dataset(split="test", **kwargs)

    def _get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.dataset_cls.collate_fn,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset)

    def _make_graph_data(self, atoms_list: List[Atoms], **kwargs) -> List[Data]:
        """Make list of torch_geometric.data.Data from list of ASE.Atoms."""
        graphs = []
        for i, atoms in enumerate(tqdm(atoms_list)):
            edge_src, edge_dst, edge_shift = neighbor_list(
                "ijS",
                a=atoms,
                cutoff=self.cutoff,
                self_interaction=True,
            )
            pos = torch.tensor(atoms.get_positions()).float()
            lattice = torch.tensor(atoms.cell.array).unsqueeze(0).float()
            edge_shift = torch.tensor(edge_shift).float()
            relative_vec = (
                pos[edge_dst]
                - pos[edge_src]
                + torch.einsum("ni,nij->nj", edge_shift, lattice)
            )
            graph_data = {
                "pos": pos,
                "lattice": lattice,
                "x": torch.tensor(atoms.get_atomic_numbers()),
                "edge_index": torch.stack(
                    [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
                ),
                "edge_shift": edge_shift,
                "relative_vec": relative_vec,
            }
            # add kwargs
            for k, v in kwargs.items():
                if isinstance(v, Iterable):
                    graph_data[k] = v[i]
                else:
                    graph_data[k] = v

            graph = Data(**graph_data)

            graphs.append(graph)
        return graphs
