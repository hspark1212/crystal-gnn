# make DGL graphs
# modified from ALIGNN (https://github.com/usnistgov/alignn/blob/main/alignn/graphs.py)
from typing import Optional, Tuple, Set
from collections import defaultdict

import numpy as np
from pymatgen.core import Structure

import dgl
import torch

from jarvis.core.atoms import Atoms


def jarvis_atoms_to_dgl_graph(
    atoms: Atoms,
    neighbor_strategy: str = "k-nearest",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    use_canonize: bool = True,
) -> dgl.DGLGraph:
    """Convert JARVIS Atoms object to DGL graph.

    Args:
        atoms (Atoms): JARVIS Atoms object
        neighbor_strategy (str, optional): neighbor strategy. Defaults to "k-nearest".
        cutoff (float, optional): cutoff distance. Defaults to 8.0.
        max_neighbors (int, optional): maximum number of neighbors. Defaults to 12.
        use_canonize (bool, optional): whether to use canonize. Defaults to True.

    Returns:
        dgl.DGLGraph: DGL graph
    """
    # get node, edge, and edge distance
    if neighbor_strategy == "k-nearest":
        # get edges with k-nearest neighbors
        edges = nearest_neighbor_edges(
            atoms,
            cutoff,
            max_neighbors,
            use_canonize,
        )
        _u, _v, _r = build_undirected_edgedata(atoms, edges)
    elif neighbor_strategy == "radius_graph":
        pass  # not supported yet
    else:
        raise ValueError(
            f"neighbor_strategy must be one of k-nearest, radius_graph, but got {neighbor_strategy}"
        )
    # construct DGL graph
    graph = dgl.graph((_u, _v))
    # add node features
    graph.ndata["volume"] = torch.tensor([atoms.volume] * atoms.num_atoms)
    graph.ndata["coord"] = torch.tensor(atoms.cart_coords)
    graph.ndata["atomic_number"] = torch.tensor(atoms.atomic_numbers).long()
    # add edge features
    graph.edata["coord_diff"] = _r

    return graph


def nearest_neighbor_edges(
    atoms: Atoms,
    cutoff: float,
    max_neighbors: int,
    use_canonize: bool,
    max_attempts: int = 3,
) -> defaultdict[Tuple[int, int], Set[Tuple[float, float, float]]]:
    """Get edges with k-nearest neighbors.

    Args:
        atoms (Atoms): JARVIS Atoms object
        cutoff (float): cutoff distance
        max_neighbors (int): maximum number of neighbors
        use_canonize (bool): whether to use canonize
        max_attempts (int, optional): maximum number of attempts to find enough neighbors.
    Returns:
        edges (defaultdict[Tuple[int, int], Set[Tuple[float, float, float]]]): edges with images
    """
    # increase cutoff radius if minimum number of neighbors is higher than max_neighbors
    attempt = 0
    while True:
        # get all neighbors within the cutoff radius
        all_neighbors = atoms.get_all_neighbors(r=cutoff)

        # find the minimum number of neighbors
        min_nbrs = min(map(len, all_neighbors))

        # if there are fewer neighbors than the maximum allowed, increase the cutoff radius
        if min_nbrs < max_neighbors:
            # Calculate the new cutoff radius
            lat = atoms.lattice
            cutoff = max([lat.a, lat.b, lat.c, 2 * cutoff])
            attempt += 1
            if attempt > max_attempts:
                raise RuntimeError(
                    "Could not find enough neighbors to satisfy max_neighbors"
                )
        else:
            break

    # get edges with distance
    def canonize_edge(
        src_id,
        dst_id,
        src_image,
        dst_image,
    ):
        """Get canonical edge representation."""
        # store directed edges src_id <= dst_id
        if dst_id < src_id:
            src_id, dst_id = dst_id, src_id
            src_image, dst_image = dst_image, src_image

        # shift periodic images so that src is in (0,0,0) image
        if not np.array_equal(src_image, (0, 0, 0)):
            shift = src_image
            src_image = tuple(np.subtract(src_image, shift))
            dst_image = tuple(np.subtract(dst_image, shift))

        assert src_image == (0, 0, 0)

        return src_id, dst_id, src_image, dst_image

    edges = defaultdict(set)
    for site_idx, neighbors in enumerate(all_neighbors):
        # sort neighbors by distance
        neighbors = sorted(neighbors, key=lambda x: x[2])
        # get distance and neighbor indices and images
        distances = np.array([nbr[2] for nbr in neighbors])
        ids = np.array([nbr[1] for nbr in neighbors])
        images = np.array([nbr[3] for nbr in neighbors])

        # get the maximum distance with k-nearest neighbors
        max_dist = distances[max_neighbors - 1]

        # keep only neighbors within the cutoff radius
        ids = ids[distances <= max_dist]
        images = images[distances <= max_dist]
        distances = distances[distances <= max_dist]
        # get edges with images
        for dst, image in zip(ids, images):
            src_id, dst_id, _, dst_image = canonize_edge(
                site_idx, dst, (0, 0, 0), tuple(image)
            )
            if use_canonize:
                edges[(src_id, dst_id)].add(dst_image)
            else:
                edges[(site_idx, dst)].add(tuple(image))

    return edges


def build_undirected_edgedata(
    atoms: Atoms,
    edges: Optional[defaultdict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build undirected graph data from edge set.
    Implemented in ALIGNN.

    Args:
        atoms (Atoms): JARVIS Atoms object
        edges (Dict): edge set with images
    """
    _u, _v, _r = [], [], []
    for (src_id, dst_id), images in edges.items():
        for dst_image in images:
            # fractional coordinate for periodic image of dst
            dst_coord = atoms.frac_coords[dst_id] + dst_image
            # cartesian displacement vector pointing from src -> dst
            dst = atoms.lattice.cart_coords(dst_coord - atoms.frac_coords[src_id])
            # add edges for both directions
            for _uu, _vv, _dd in [(src_id, dst_id, dst), (dst_id, src_id, -dst)]:
                _u.append(_uu)
                _v.append(_vv)
                _r.append(_dd)
    _u, _v, _r = (np.array(x) for x in (_u, _v, _r))
    _u = torch.tensor(_u)
    _v = torch.tensor(_v)
    _r = torch.tensor(_r).type(torch.get_default_dtype())

    return _u, _v, _r


def compute_bond_cosines(edges):
    """Compute bond angle cosines from bond displacement vectors.
    from jarvis.core.graphs import compute_bond_cosines
    """
    r1 = -edges.src["coord_diff"]
    r2 = edges.dst["coord_diff"]
    bond_cosine = torch.sum(r1 * r2, dim=1) / (
        torch.norm(r1, dim=1) * torch.norm(r2, dim=1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return {"angle": bond_cosine}  # a is edge features (bond angle cosines)


def convert_structures_to_jarvis_atoms(structure: Structure) -> Atoms:
    return Atoms(
        lattice_mat=structure.lattice.matrix,
        coords=structure.frac_coords,
        elements=[i.symbol for i in structure.species],
        cartesian=False,
    )
