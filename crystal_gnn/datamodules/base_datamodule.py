from typing import Dict, Any

from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader

from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(self, _config: Dict[str, Any]) -> None:
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        # configs for dataset
        self.compute_line_graph = _config["compute_line_graph"]
        self.neighbor_strategy = _config["neighbor_strategy"]
        self.cutoff = _config["cutoff"]
        self.max_neighbors = _config["max_neighbors"]
        self.use_canonize = _config["use_canonize"]
        # configs for dataloader
        self.batch_size = _config["batch_size"]
        self.num_workers = _config["num_workers"]
        self.pin_memory = _config["pin_memory"]
        self.use_ddp = _config["use_ddp"]

    @property
    def dataset_cls(self) -> DGLDataset:
        raise NotImplementedError

    @property
    def dataset_name(self) -> str:
        raise NotImplementedError

    def _set_dataset(self, split: str) -> DGLDataset:
        print(f"Start setting {split} dataset...")
        return self.dataset_cls(
            database_name=self.database_name,
            target=self.target,
            split=split,
            compute_line_graph=self.compute_line_graph,
            data_dir=self.data_dir,
            neighbor_strategy=self.neighbor_strategy,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            use_canonize=self.use_canonize,
        )

    def setup(self, stage: str = None) -> None:
        # train
        if stage == "fit":
            self.train_dataset = self._set_dataset(split="train")
            self.val_dataset = self._set_dataset(split="val")
        # test
        elif stage == "test":
            self.test_dataset = self._set_dataset(split="test")
        # predict
        elif stage == "predict":
            self.test_dataset = self._set_dataset(split="test")
        else:
            self.train_dataset = self._set_dataset(split="train")
            self.val_dataset = self._set_dataset(split="val")
            self.test_dataset = self._set_dataset(split="test")

    def _get_dataloader(
        self, dataset: DGLDataset, shuffle: bool = False
    ) -> GraphDataLoader:
        return GraphDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.dataset_cls.collate_fn,
            use_ddp=self.use_ddp,
            drop_last=False,
        )

    def train_dataloader(self) -> GraphDataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> GraphDataLoader:
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> GraphDataLoader:
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> GraphDataLoader:
        return self._get_dataloader(self.test_dataset)
