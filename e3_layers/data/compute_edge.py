import torch
from .data import Data
import warnings
import numpy as np
import ase.neighborlist
import ase
from tqdm import tqdm, trange


def neighbor_list_and_relative_vec(
    pos,
    r_max,
    self_interaction=False,
    strict_self_interaction=True,
    cell=None,
    pbc=False,
):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    If the input positions are a tensor with ``requires_grad == True``,
    the output displacement vectors will be correctly attached to the inputs
    for autograd.

    All outputs are Tensors on the same device as ``pos``; this allows future
    optimization of the neighbor list on the GPU.

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        r_max (float): Radial cutoff distance for neighbor finding.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Whether the system is periodic in each of the three cell dimensions.
        self_interaction (bool): Whether or not to include same periodic image self-edges in the neighbor list.
        strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the two
            instances of the atom are in different periodic images. Defaults to True, should be True for most applications.

    Returns:
        edge_index (torch.tensor shape [2, num_edges]): List of edges.
        edge_cell_shift (torch.tensor shape [num_edges, 3]): Relative cell shift
            vectors. Returned only if cell is not None.
        cell (torch.Tensor [3, 3]): the cell as a tensor on the correct device.
            Returned only if cell is not None.
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # Either the position or the cell may be on the GPU as tensors
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    # Right now, GPU tensors require a round trip
    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. Please pass CPU tensors if possible."
        )

    # Get a cell on the CPU no matter what
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
        cell_tensor = cell.to(device=out_device, dtype=out_dtype)
    elif cell is not None:
        temp_cell = np.asarray(cell)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)
    else:
        # ASE will "complete" this correctly.
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)
        cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    # ASE dependent part
    temp_cell = ase.geometry.complete_cell(temp_cell)

    first_idex, second_idex, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=float(r_max),
        self_interaction=strict_self_interaction,  # we want edges from atom to itself in different periodic images!
        use_scaled_positions=False,
    )

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        bad_edge &= np.all(shifts == 0, axis=1)
        keep_edge = ~bad_edge
        if not np.any(keep_edge):
            raise ValueError(
                "After eliminating self edges, no edges remain in this system."
            )
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        shifts = shifts[keep_edge]

    # Build output:
    edge_index = torch.vstack(
        (torch.LongTensor(first_idex), torch.LongTensor(second_idex))
    ).to(device=out_device)

    shifts = torch.as_tensor(
        shifts,
        dtype=out_dtype,
        device=out_device,
    )
    return edge_index, shifts, cell_tensor


def computeEdgeVector(data: Data, with_lengths: bool = True):
    """Compute the edge displacement vectors for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 3] edge displacement vectors
    """
    data.attrs["edge_vector"] = ("edge", "1x1o")
    data.attrs["edge_length"] = ("edge", "1x0e")
    if "edge_vector" in data.keys():
        if with_lengths and "edge_length" not in data.keys():
            data["edge_length"] = torch.linalg.norm(data["edge_vector"], dim=-1)
        return data
    else:
        # Build it dynamically
        # Note that this is
        # (1) backwardable, because everything (pos, cell, shifts)
        #     is Tensors.
        # (2) works on a Batch constructed from AtomicData
        pos = data["pos"]
        edge_index = data["edge_index"]
        edge_vec = pos[edge_index[1]] - pos[edge_index[0]]
        if "cell" in data:
            # ^ note that to save time we don't check that the edge_cell_shifts are trivial if no cell is provided; we just assume they are either not present or all zero.
            # -1 gives a batch dim no matter what
            cell = data["cell"].view(-1, 3, 3)
            edge_cell_shift = data["edge_cell_shift"]
            if cell.shape[0] > 1:
                batch = data["batch"]
                # Cell has a batch dimension
                # note the ASE cell vectors as rows convention
                edge_vec = edge_vec + torch.einsum(
                    "ni,nij->nj", edge_cell_shift, cell[batch[edge_index[0]]]
                )
                # TODO: is there a more efficient way to do the above without
                # creating an [n_edge] and [n_edge, 3, 3] tensor?
            else:
                # Cell has either no batch dimension, or a useless one,
                # so we can avoid creating the large intermediate cell tensor.
                # Note that we do NOT check that the batch array, if it is present,
                # is trivial — but this does need to be consistent.
                edge_vec = edge_vec + torch.einsum(
                    "ni,ij->nj",
                    edge_cell_shift,
                    cell.squeeze(0),  # remove batch dimension
                )
        data["edge_vector"] = edge_vec
        if with_lengths:
            data["edge_length"] = torch.linalg.norm(edge_vec, dim=-1)
        return data


def computeEdgeIndex(
    batch,
    r_max: float = None,
    strict_self_interaction: bool = True,
    **kwargs
):
    """Build neighbor graph from points, optionally with PBC.

    Args:
        pos (np.ndarray/torch.Tensor shape [N, 3]): node positions. If Tensor, must be on the CPU.
        r_max (float): neighbor cutoff radius.
    """

    pos = batch["pos"]
    pos = torch.as_tensor(pos, dtype=torch.get_default_dtype())

    lst = []
    cnt = 0
    if '_n_nodes' in batch:
        for i, n in enumerate(batch['_n_nodes']):
            tmp = pos[cnt:cnt+n]
            edge_index, _, _ = neighbor_list_and_relative_vec(
                    pos=tmp,
                    r_max=r_max,
                    self_interaction=False,
                    strict_self_interaction=strict_self_interaction
                )
            cnt += n
            lst.append(edge_index)
        edge_index = torch.cat(lst, dim=-1)
    else:
        edge_index, _, _ = neighbor_list_and_relative_vec(
                pos=pos,
                r_max=r_max,
                self_interaction=False,
                strict_self_interaction=strict_self_interaction
            )
    edge_index = edge_index.to(pos.device)

    attrs = batch.attrs
    attrs["_n_edges"] = ('graph', '1x0e')
    batch["edge_index"] = edge_index

    return batch
