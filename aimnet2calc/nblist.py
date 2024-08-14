import torch
from torch import Tensor
from typing import Optional, Tuple
from torch_cluster import radius_graph, radius
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


def nblists_torch_pbc(coord: Tensor, cell: Tensor, cutoff: float, max_nb: int=48) -> Tuple[Tensor, Tensor, Tensor]:
    """ Compute dense neighbor lists for periodic boundary conditions case.
    Coordinates must be in cartesian coordinates and be within the unit cell.
    Single crystal only, no support for batched coord or multiple unit cells.
    """
    assert coord.ndim == 2, 'Expected 2D tensor for coord, got {coord.ndim}D'
    device = coord.device
    
    reciprocal_cell = cell.inverse().t()
    inv_distances = reciprocal_cell.norm(2, -1)
    shifts = _calc_shifts(inv_distances, cutoff)
    
    if coord.shape[0] > 10e3 and cutoff < 20: # avoid making big NxMxS conn_mat for big systems
        _fn = nblist_torch_cluster_pbc
        mat_idxj, mat_pad, mat_S = _fn(coord, cell, shifts, cutoff, max_nb)
        return mat_idxj, mat_pad, mat_S
    
    d = torch.cdist(coord.unsqueeze(0), coord.unsqueeze(0) + (shifts @ cell).unsqueeze(1))
    conn_mat = ((d < cutoff) & (d > 0.1)).transpose(0, 1).contiguous()
    
    if device.type == 'cuda' and _numba_cuda_available:
        _fn = _nblist_pbc_cuda
        mat_idxj, mat_pad, mat_S = _fn(conn_mat, shifts)
    else:
        _fn = _nblist_pbc_cpu
        mat_idxj, mat_pad, mat_S = _fn(conn_mat, shifts, device)
    return mat_idxj, mat_pad, mat_S


def nblist_torch_cluster_pbc(coord: Tensor, cell: Tensor, shifts: Tensor,
                             cutoff: float, max_nb: int) -> Tuple[Tensor, Tensor, Tensor]:
    assert coord.ndim == 2, 'Expected 2D tensor for coord, got {coord.ndim}D'
    device = coord.device
    
    # put the zero shift first for convenience when removing self-interaction from torch_cluster.radius
    ind = torch.argwhere(torch.all(shifts == 0, axis=1))[0,0]
    shifts = shifts[torch.tensor([ind] +  \
        list(range(ind)) + \
            list(range(ind+1, len(shifts))))].clone().detach().to(device).to(torch.int8)
    
    supercoord = torch.vstack(
        [coord+(shift @ cell) for shift in shifts.to(torch.float)])
    
    max_num_neighbors = max_nb
    flag = True
    while flag:
        edges = radius(supercoord, coord, cutoff, max_num_neighbors=max_nb)
        max_num_neighbors = torch.unique(edges[0], return_counts=True)[1].max().item()
        flag = max_num_neighbors == max_nb
        if flag:
            max_nb = int(max_nb * 1.5)

    orig_len = coord.shape[0]
    mat_idxj = torch.full((orig_len+1, max_nb), orig_len, dtype=torch.int).to(device)
    mat_pad = torch.ones((orig_len+1, max_nb), dtype=torch.int8).to(device)
    mat_S = torch.full((orig_len+1, max_nb, 3), -1, dtype=torch.int8).to(device)
    
    if device.type == 'cuda':
        threadsperblock = 32
        blockspergrid = (edges.shape[1] + (threadsperblock - 1)) // threadsperblock
        _nblist_sparse_pbc_cuda[blockspergrid, threadsperblock](
                                                        orig_len, edges, shifts,
                                                        numba.cuda.as_cuda_array(mat_idxj),
                                                        numba.cuda.as_cuda_array(mat_pad),
                                                        numba.cuda.as_cuda_array(mat_S))
        return mat_idxj, mat_pad, mat_S
        
    else:
        mat_idxj = mat_idxj.cpu().numpy(); mat_pad = mat_pad.cpu().numpy(); mat_S = mat_S.cpu().numpy()
        _nblist_sparse_pbc(orig_len, edges.cpu().numpy(), shifts.cpu().numpy(),
                                mat_idxj, mat_pad, mat_S)
        return torch.tensor(mat_idxj).to(device), torch.tensor(mat_pad).to(device), torch.tensor(mat_S).to(device)
    
@numba.njit(cache=True)
def _nblist_sparse_pbc(orig_len, edges, shifts, nl, nl_pad, nl_shifts):
    e0 = edges[0]
    e1 = edges[1]
    mask = e0 != e1
    e0 = e0[mask]
    e1 = e1[mask]
    e1r = e1 % orig_len
    e1d = e1 // orig_len
    prev = -1
    tmp = 0
    for ct, i in enumerate(e0):
        if i == prev:
            tmp += 1
        else:
            tmp = 0
        prev = i
        nl[i, tmp] = e1r[ct]
        nl_pad[i, tmp] = 0
        nl_shifts[i, tmp] = shifts[e1d[ct]]
    return nl, nl_pad, nl_shifts


@numba.cuda.jit(cache=True)
def _nblist_sparse_pbc_cuda(orig_len, edges, shifts, nl, nl_pad, nl_shifts):
    e0 = edges[0]
    e1 = edges[1]
    gi = numba.cuda.grid(1)
    if gi > e0[-1]:
        return
    
    #initialise position in array
    nn = e0.shape[0]//e0[-1]
    start = gi*nn
    if e0[start] < gi:
        while e0[start] < gi:
            start += 1
    else:
        while start > 0:
            if e0[start] >= gi:
                start -= 1
            else:
                start += 1
                break
    
    # start assigning
    tmp = 0
    while e0[start] == gi and start<e0.shape[0]:
        if e0[start] == e1[start]:
            start += 1
            continue
        nl[gi, tmp] = e1[start] % orig_len
        nl_pad[gi, tmp] = 0
        for j in range(3):
            nl_shifts[gi, tmp, j] = shifts[e1[start] // orig_len, j]
        start += 1
        tmp += 1


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
