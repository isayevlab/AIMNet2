from typing import Optional, Tuple
import torch
from torch import Tensor
import numba
import warnings

if torch.cuda.is_available():
    import numba.cuda
    assert numba.cuda.is_available(), "PyTorch CUDA is available, but Numba CUDA is not available."
    _numba_cuda_available = True
else:
    _numba_cuda_available = False


@numba.njit(cache=True, parallel=True)
def _nbmat_cpu(coord, cutoff_squared, maxnb, mol_idx, mol_end_idx, nbmat, nnb):
    # number of atoms
    N = coord.shape[0]
    # parallel loop over atoms
    for i in numba.prange(N):
        # coordinates of atom i
        c_i = coord[i]
        # mol index of atom i
        _mol_idx = mol_idx[i]
        # get indices of other atoms within same mol with j>i
        _j_start = i + 1
        _j_end = mol_end_idx[_mol_idx]
        # loop over other atoms in mol
        for j in range(_j_start, _j_end):
            c_j = coord[j]
            diff = c_i - c_j
            dist2 = (diff * diff).sum(-1)
            if dist2 < cutoff_squared:
                pos = nnb[i]
                nnb[i] += 1
                if pos < maxnb:
                    nbmat[i, pos] = j
    # add pairs with j<i
    nnb_half = nnb.copy()
    for i in range(N):
        for m in range(nnb_half[i]):
            j = nbmat[i, m]
            pos = nnb[j]
            nnb[j] += 1
            if pos < maxnb:
                nbmat[j, pos] = i


@numba.njit(cache=True, parallel=True)
def _nbmat_dual_cpu(coord, cutoff1_squared, cutoff2_squared, maxnb1, maxnb2, mol_idx, mol_end_idx, nbmat1, nbmat2, nnb1, nnb2):
    # dual cutoff version of _nbmat_cpu
    N = coord.shape[0]
    for i in numba.prange(N):
        c_i = coord[i]
        _mol_idx = mol_idx[i]
        _j_start = i + 1
        _j_end = mol_end_idx[_mol_idx]
        for j in range(_j_start, _j_end):
            c_j = coord[j]
            diff = c_i - c_j
            dist2 = (diff * diff).sum(-1)
            if dist2 < cutoff1_squared:
                pos = nnb1[i]
                nnb1[i] += 1
                if pos < maxnb1:
                    nbmat1[i, pos] = j
            if dist2 < cutoff2_squared:
                pos = nnb2[i]
                nnb2[i] += 1
                if pos < maxnb2:
                    nbmat2[i, pos] = j
    nnb1_half = nnb1.copy()
    nnb2_half = nnb2.copy()
    for i in range(N):
        for m in range(nnb1_half[i]):
            j = nbmat1[i, m]
            pos = nnb1[j]
            nnb1[j] += 1
            if pos < maxnb1:
                nbmat1[j, pos] = i
        for m in range(nnb2_half[i]):
            j = nbmat2[i, m]
            pos = nnb2[j]
            nnb2[j] += 1
            if pos < maxnb2:
                nbmat2[j, pos] = i


def _nbmat_cuda(coord, cutoff_squared, maxnb, mol_idx, mol_end_idx, nbmat, nnb):
    N = coord.shape[0]
    i = numba.cuda.grid(1)

    if (i >= N):
        return    

    c0 = coord[i, 0]
    c1 = coord[i, 1]
    c2 = coord[i, 2]

    _mol_idx = mol_idx[i]
    _j_start = i + 1
    _j_end = mol_end_idx[_mol_idx]

    for j in range(_j_start, _j_end):
        d0 = c0 - coord[j, 0]
        d1 = c1 - coord[j, 1]
        d2 = c2 - coord[j, 2]
        dist_squared = d0 * d0 + d1 * d1 + d2 * d2
        if dist_squared > cutoff_squared:
            continue

        pos = numba.cuda.atomic.add(nnb, i, 1)
        if pos < maxnb:
            nbmat[i, pos] = j
        pos = numba.cuda.atomic.add(nnb, j, 1)
        if pos < maxnb:
            nbmat[j, pos] = i


def _nbmat_dual_cuda(coord, cutoff1_squared, cutoff2_squared, maxnb1, maxnb2, mol_idx, mol_end_idx, nbmat1, nbmat2, nnb1, nnb2):
    N = coord.shape[0]
    i = numba.cuda.grid(1)
    
    if (i >= N):
        return

    c0 = coord[i, 0]
    c1 = coord[i, 1]
    c2 = coord[i, 2]

    _mol_idx = mol_idx[i]
    _j_start = i + 1
    _j_end = mol_end_idx[_mol_idx]

    for j in range(_j_start, _j_end):
        d0 = c0 - coord[j, 0]
        d1 = c1 - coord[j, 1]
        d2 = c2 - coord[j, 2]
        dist_squared = d0 * d0 + d1 * d1 + d2 * d2
        if dist_squared < cutoff1_squared:
            pos = numba.cuda.atomic.add(nnb1, i, 1)
            if pos < maxnb1:
                nbmat1[i, pos] = j
            pos = numba.cuda.atomic.add(nnb1, j, 1)
            if pos < maxnb1:
                nbmat1[j, pos] = i
        if dist_squared < cutoff2_squared:
            pos = numba.cuda.atomic.add(nnb2, i, 1)
            if pos < maxnb2:
                nbmat2[i, pos] = j
            pos = numba.cuda.atomic.add(nnb2, j, 1)
            if pos < maxnb2:
                nbmat2[j, pos] = i


@numba.njit(cache=True, parallel=True)
def _nbmat_pbc_cpu(coord, coord_shifted, cutoff_squared, maxnb, nbmat, nnb, nbmat_shifts):
    N = coord.shape[0]
    M = coord_shifted.shape[0]
    for i in numba.prange(N):
        c_i = coord[i]
        for j in range(M):
            c_j = coord_shifted[j]
            diff = c_i - c_j
            dist2 = (diff * diff).sum(-1)
            if dist2 > 0.01 and dist2 < cutoff_squared:
                pos = nnb[i]
                nnb[i] += 1
                if pos < maxnb:
                    nbmat[i, pos] = j % N
                    nbmat_shifts[i, pos] = j // N


def _nbmat_pbc_cuda(coord, coord_shifted, cutoff_squared, maxnb, nbmat, nnb, nbmat_shifts):
    N = coord_shifted.shape[0]
    M = coord.shape[0]

    i = numba.cuda.grid(1)
    if i >= N:
        return

    c0 = coord_shifted[i, 0]
    c1 = coord_shifted[i, 1]
    c2 = coord_shifted[i, 2]

    k = i % M
    l = i // M

    for j in range(M):
        d0 = c0 - coord[j, 0]
        d1 = c1 - coord[j, 1]
        d2 = c2 - coord[j, 2]
        dist_squared = d0 * d0 + d1 * d1 + d2 * d2
        if dist_squared > 0.01 and dist_squared < cutoff_squared:
            pos = numba.cuda.atomic.add(nnb, j, 1)
            if pos < maxnb:
                nbmat[j, pos] = k
                nbmat_shifts[j, pos] = l


if _numba_cuda_available:
    numba_cuda_jit_kwargs = {'fastmath': True, 'cache': True}
    _nbmat_cuda = numba.cuda.jit(_nbmat_cuda, **numba_cuda_jit_kwargs)
    _nbmat_dual_cuda = numba.cuda.jit(_nbmat_dual_cuda, **numba_cuda_jit_kwargs)
    _nbmat_pbc_cuda = numba.cuda.jit(_nbmat_pbc_cuda, **numba_cuda_jit_kwargs)


def calc_nbmat_dual(coord: Tensor,
                cutoffs: Tuple[float, Optional[float]],
                maxnb: Tuple[int, Optional[int]],
                mol_idx: Optional[Tensor] = None,
                ):
    device = coord.device
    N = coord.shape[0]

    threadsperblock = 32
    blockspergrid = (N + (threadsperblock - 1)) // threadsperblock

    cutoff_sr, cutoff_lr = cutoffs
    maxnb_sr, maxnb_lr = maxnb

    _, mol_size = torch.unique(mol_idx, return_counts=True)
    mol_end_idx = torch.cumsum(mol_size, 0)

    if cutoff_lr is None:
        nnb = torch.zeros(N, dtype=torch.long, device=device)
        nbmat = torch.full((N+1, maxnb_sr), N, dtype=torch.long, device=device)
        cutoff_squared = cutoff_sr * cutoff_sr
        if device.type == 'cuda':
            fn = _nbmat_cuda[blockspergrid, threadsperblock]
            _coord = numba.cuda.as_cuda_array(coord)
            _nbmat1 = numba.cuda.as_cuda_array(nbmat)
            _nnb1 = numba.cuda.as_cuda_array(nnb)
            _mol_idx = numba.cuda.as_cuda_array(mol_idx)
            _mol_end_idx = numba.cuda.as_cuda_array(mol_end_idx)
        else:
            fn = _nbmat_cpu
            _coord = coord.numpy()
            _nbmat1 = nbmat.numpy()
            _nnb1 = nnb.numpy()
            _mol_idx = mol_idx.numpy()
            _mol_end_idx = mol_end_idx.numpy()

        fn(_coord, cutoff_squared, maxnb_sr, _mol_idx, _mol_end_idx, _nbmat1, _nnb1)
        nnb1_max = nnb.max()
        if nnb1_max > maxnb_sr:
            raise ValueError(f"Max number of neighbors exceeded, increase maxnb_sr.")
        nbmat1 = torch.as_tensor(nbmat1[:, :nnb1_max], device=device)
        nbmat2 = None
    
    else:
        nnb1 = torch.zeros(N, dtype=torch.long, device=device)
        nnb2 = torch.zeros(N, dtype=torch.long, device=device)
        nbmat1 = torch.full((N+1, maxnb_sr), N, dtype=torch.long, device=device)
        nbmat2 = torch.full((N+1, maxnb_lr), N, dtype=torch.long, device=device)
        cutoff1_squared = cutoff_sr * cutoff_sr
        cutoff2_squared = cutoff_lr * cutoff_lr
        if device.type == 'cuda' and _numba_cuda_available:
            fn = _nbmat_dual_cuda[blockspergrid, threadsperblock]
            _coord = numba.cuda.as_cuda_array(coord)
            _nbmat1 = numba.cuda.as_cuda_array(nbmat1)
            _nbmat2 = numba.cuda.as_cuda_array(nbmat2)
            _nnb1 = numba.cuda.as_cuda_array(nnb1)
            _nnb2 = numba.cuda.as_cuda_array(nnb2)
            _mol_idx = numba.cuda.as_cuda_array(mol_idx)
            _mol_end_idx = numba.cuda.as_cuda_array(mol_end_idx)
        else:
            fn = _nbmat_dual_cpu
            _coord = coord.numpy()
            _nbmat1 = nbmat1.numpy()
            _nbmat2 = nbmat2.numpy()
            _nnb1 = nnb1.numpy()
            _nnb2 = nnb2.numpy()
            _mol_idx = mol_idx.numpy()
            _mol_end_idx = mol_end_idx.numpy()
            
        fn(_coord, cutoff1_squared, cutoff2_squared, maxnb_sr, maxnb_lr, _mol_idx, _mol_end_idx, _nbmat1, _nbmat2, _nnb1, _nnb2)
        nnb1_max = nnb1.max()
        nnb2_max = nnb2.max()
        if nnb1_max > maxnb_sr:
            raise ValueError(f"Max number of neighbors exceeded, increase maxnb_sr.")
        if nnb2_max > maxnb_lr:
            raise ValueError(f"Max number of neighbors exceeded, increase maxnb_lr.")
        nbmat1 = nbmat1[:, :nnb1_max]
        nbmat2 = nbmat2[:, :nnb2_max]
    
    return nbmat1, nbmat2


def calc_nbmat_pbc(coord: Tensor,
                   cell: Tensor,
                   cutoff: float,
                   maxnb: int
                   ):
    device = coord.device

    inv_distances = cell.detach().inverse().cpu().norm(2, -1)
    nshifts = torch.ceil(cutoff * inv_distances).to(torch.long)
    dc = [torch.arange(-nshifts[i], nshifts[i] + 1) for i in range(len(nshifts))]
    shifts = torch.cartesian_prod(*dc).to(torch.float).to(device)
    coord_shifted = coord.unsqueeze(0) + (shifts @ cell).unsqueeze(1)
    ncells = shifts.shape[0]
    shifts = torch.nn.functional.pad(shifts, [0, 0, 0, 1], mode='constant', value=0.0)
    coord_shifted = coord_shifted.view(-1, 3).contiguous()

    N = coord_shifted.shape[0]
    threadsperblock = 32
    blockspergrid = (N + (threadsperblock - 1)) // threadsperblock

    M = coord.shape[0]
    nnb = torch.zeros(M, dtype=torch.long, device=device)
    nbmat = torch.full((M+1, maxnb), M, dtype=torch.long, device=device)
    nbmat_shifts = torch.full((M+1, maxnb), ncells, dtype=torch.long, device=device)
    cutoff_squared = cutoff * cutoff

    if device.type == 'cuda' and _numba_cuda_available:
        fn = _nbmat_pbc_cuda[blockspergrid, threadsperblock]
        _coord = numba.cuda.as_cuda_array(coord)
        _coord_shifted = numba.cuda.as_cuda_array(coord_shifted)
        _nbmat = numba.cuda.as_cuda_array(nbmat)
        _nnb = numba.cuda.as_cuda_array(nnb)
        _nbmat_shifts = numba.cuda.as_cuda_array(nbmat_shifts)
    else:
        fn = _nbmat_pbc_cpu
        _coord = coord.numpy()
        _coord_shifted = coord_shifted.numpy()
        _nbmat = nbmat.numpy()
        _nnb = nnb.numpy()
        _nbmat_shifts = nbmat_shifts.numpy()
    fn(_coord, _coord_shifted, cutoff_squared, maxnb, _nbmat, _nnb, _nbmat_shifts)
    nnb_max = nnb.max()
    if nnb_max > maxnb:
        raise ValueError(f"Max number of neighbors exceeded, increase maxnb.")
    nbmat = nbmat[:, :nnb_max]
    nbmat_shifts = nbmat_shifts[:, :nnb_max]
    shifts = shifts[nbmat_shifts]
    return nbmat, shifts
