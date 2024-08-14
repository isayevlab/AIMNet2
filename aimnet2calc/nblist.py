import torch
from torch import Tensor
from typing import Optional, Tuple
from torch_cluster import radius_graph
import numba
try:
    # optionaly use numba cuda
    import numba.cuda
    _numba_cuda_available = True
except ImportError:
    _numba_cuda_available = False
import numpy as np


@numba.njit(cache=True)
def sparse_nb_to_dense_half(idx, natom, max_nb):
    dense_nb = np.full((natom+1, max_nb), natom, dtype=np.int32)
    last_idx = np.zeros((natom,), dtype=np.int32)
    for k in range(idx.shape[0]):
        i, j = idx[k]
        il, jl = last_idx[i], last_idx[j]
        dense_nb[i, il] = j
        dense_nb[j, jl] = i
        last_idx[i] += 1
        last_idx[j] += 1
    return dense_nb


def nblist_torch_cluster(coord: Tensor, cutoff: float, mol_idx: Optional[Tensor] = None, max_nb: int = 256):
    device = coord.device
    assert coord.ndim == 2, 'Expected 2D tensor for coord, got {coord.ndim}D'
    assert coord.shape[0] < 2147483646, 'Too many atoms, max supported is 2147483646'
    max_num_neighbors = max_nb
    while True:
        sparse_nb = radius_graph(coord, batch=mol_idx, r=cutoff, max_num_neighbors=max_nb).to(torch.int32)
        nnb = torch.unique(sparse_nb[0], return_counts=True)[1]
        if nnb.numel() == 0:
            break
        max_num_neighbors = nnb.max().item()
        if max_num_neighbors < max_nb:
            break
        max_nb *= 2
    sparse_nb_half = sparse_nb[:, sparse_nb[0] > sparse_nb[1]]
    dense_nb = sparse_nb_to_dense_half(sparse_nb_half.mT.cpu().numpy(), coord.shape[0], max_num_neighbors)
    dense_nb = torch.as_tensor(dense_nb, device=device)
    return dense_nb


### dense neighbor matrix kernels

@numba.njit(cache=True, parallel=True)
def _cpu_dense_nb_mat_sft(conn_matrix):
    N, S = conn_matrix.shape[:2]
    # figure out max number of neighbors
    _s_flat_conn_matrix = conn_matrix.reshape(N, -1)
    maxnb = np.max(np.sum(_s_flat_conn_matrix, axis=-1))
    M = maxnb
    # atom idx matrix
    mat_idxj = np.full((N + 1, M), N, dtype=np.int_)
    # padding matrix
    mat_pad = np.ones((N + 1, M), dtype=np.bool_)
    # shitfs matrix
    mat_S_idx = np.zeros((N + 1, M), dtype=np.int_)
    for _n in numba.prange(N):
        _i = 0
        for _s in range(S):
            for _m in range(N):
                if conn_matrix[_n, _s, _m] == True:
                    mat_idxj[_n, _i] = _m
                    mat_pad[_n, _i] = False
                    mat_S_idx[_n, _i] = _s
                    _i += 1
    return mat_idxj, mat_pad, mat_S_idx

if _numba_cuda_available:
    @numba.cuda.jit(cache=True)
    def _cuda_dense_nb_mat_sft(conn_matrix, mat_idxj, mat_pad, mat_S_idx):
        i = numba.cuda.grid(1)
        if i < conn_matrix.shape[0]:
            k = 0
            for s in range(conn_matrix.shape[1]):
                for j in range(conn_matrix.shape[2]):
                    if conn_matrix[i, s, j] > 0:
                        mat_idxj[i, k] = j
                        mat_pad[i, k] = 0
                        mat_S_idx[i, k] = s
                        k += 1


def nblists_torch_pbc(coord: Tensor, cell: Tensor, cutoff: float) -> Tuple[Tensor, Tensor, Tensor]:
    """ Compute dense neighbor lists for periodic boundary conditions case.
    Coordinates must be in cartesian coordinates and be within the unit cell.
    Single crystal only, no support for batched coord or multiple unit cells.
    """
    assert coord.ndim == 2, 'Expected 2D tensor for coord, got {coord.ndim}D'
    # non-PBC version
    device = coord.device

    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    shifts = _calc_shifts(inv_distances, cutoff)
    d = torch.cdist(coord.unsqueeze(0), coord.unsqueeze(0) + (shifts @ cell).unsqueeze(1))
    conn_mat = ((d < cutoff) & (d > 0.1)).transpose(0, 1).contiguous()
    if device.type == 'cuda' and _numba_cuda_available:
        _fn = _nblist_pbc_cuda
    else:
        _fn = _nblist_pbc_cpu
    mat_idxj, mat_pad, mat_S = _fn(conn_mat, shifts)
    return mat_idxj, mat_pad, mat_S


def _calc_shifts(inv_distances, cutoff):
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    dc = [torch.arange(-num_repeats[i], num_repeats[i] + 1, device=inv_distances.device) for i in range(len(num_repeats))]
    shifts = torch.cartesian_prod(*dc).to(torch.float)
    return shifts            


def _nblist_pbc_cuda(conn_mat, shifts):
    N = conn_mat.shape[0]
    M = conn_mat.view(N, -1).sum(-1).max()
    threadsperblock = 32
    blockspergrid = (N + (threadsperblock - 1)) // threadsperblock
    idx_j = torch.full((N + 1, M), N, dtype=torch.int64, device=conn_mat.device)
    mat_pad = torch.ones((N + 1, M), dtype=torch.int8, device=conn_mat.device)
    S_idx = torch.zeros((N + 1, M), dtype=torch.int64, device=conn_mat.device)
    conn_mat = conn_mat.to(torch.int8)
    _conn_mat = numba.cuda.as_cuda_array(conn_mat)
    _idx_j = numba.cuda.as_cuda_array(idx_j)
    _mat_pad = numba.cuda.as_cuda_array(mat_pad)
    _S_idx = numba.cuda.as_cuda_array(S_idx)
    _cuda_dense_nb_mat_sft[blockspergrid, threadsperblock](_conn_mat, _idx_j, _mat_pad, _S_idx)
    mat_pad = mat_pad.to(torch.bool)
    return idx_j, mat_pad, shifts[S_idx]


def _nblist_pbc_cpu(conn_mat, shifts, device):
    conn_mat = conn_mat.cpu().numpy()
    mat_idxj, mat_pad, mat_S_idx = _cpu_dense_nb_mat_sft(conn_mat)
    mat_idxj = torch.from_numpy(mat_idxj).to(device)
    mat_pad = torch.from_numpy(mat_pad).to(device)
    mat_S_idx = torch.from_numpy(mat_S_idx).to(device)
    mat_S = shifts[mat_S_idx]
    return mat_idxj, mat_pad, mat_S

    


            
    





