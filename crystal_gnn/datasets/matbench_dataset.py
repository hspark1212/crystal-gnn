from typing import Dict, List, Any
from tqdm import tqdm

from pymatgen.core import Structure

import dgl
from dgl.data import DGLDataset
import torch


from crystal_gnn.datasets.utils_jarvis import (
    jarvis_atoms_to_dgl_graph,
    compute_bond_cosines,
    convert_structures_to_jarvis_atoms,
)


class MatbenchDataset(DGLDataset):
    def __init__(
        self,
        names: List[str],
        structures: List[Structure],
        targets: List[Any],
        split: str,
        compute_line_graph: bool = True,
        neighbor_strategy: str = "k-nearest",
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        use_canonize: bool = True,
    ):
        """Generate MatbenchDataset.

        Args:
            names (List[str]): a list of names
            structures (List[Structure]): a list of pymatgen Structure objects
            targets (List[Any]): a list of targets
            split (str): split of the dataset, one of train, val, test
            compute_line_graph (bool, optional): compute line graph. Defaults to False.
            neighbor_strategy (str, optional): neighbor strategy. Defaults to "k-nearest".
            cutoff (float, optional): cutoff distance. Defaults to 8.0.
            max_neighbors (int, optional): maximum number of neighbors. Defaults to 12.
            use_canonize (bool, optional): whether to use canonize. Defaults to True.
        """
        super(MatbenchDataset, self).__init__(name="matbench")
        assert split in [
            "train",
            "val",
            "test",
        ], "split must be one of train, val, test"
        self.compute_line_graph = compute_line_graph
        # convert pymatgen Structures to dgl graphs
        graphs = []
        line_graphs = []
        for structure in tqdm(structures):
            atoms = convert_structures_to_jarvis_atoms(structure)
            graph = jarvis_atoms_to_dgl_graph(
                atoms,
                neighbor_strategy,
                cutoff,
                max_neighbors,
                use_canonize,
            )
            graph.apply_edges(
                lambda edges: {"distance": torch.norm(edges.data["coord_diff"], dim=1)}
            )
            graphs.append(graph)
            if compute_line_graph:
                line_graph = graph.line_graph(shared=True)
                line_graph.apply_edges(compute_bond_cosines)
                line_graphs.append(line_graph)
        self.graphs = graphs
        self.line_graphs = line_graphs
        # get targets
        self.targets = targets
        # get names
        self.names = names

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = dict()
        graph = self.graphs[index]
        target = self.targets[index]
        data.update({"graph": graph, "target": target})
        if self.compute_line_graph:
            line_graph = self.line_graphs[index]
            data.update({"line_graph": line_graph})
        return data

    @staticmethod
    def collate_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        """batch collate function for JarvisDataset."""
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        # get batch graph
        dict_batch["graph"] = dgl.batch(dict_batch["graph"])
        # get batch line graph
        if "line_graph" in keys:
            dict_batch["line_graph"] = dgl.batch(dict_batch["line_graph"])
        # get batch target
        dict_batch["target"] = torch.tensor(dict_batch["target"])
        return dict_batch
