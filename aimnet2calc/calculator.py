import torch
from torch import nn, Tensor
from typing import Union, Dict, Any
from aimnet2calc.nblist import nblist_torch_cluster, nblists_torch_pbc
from aimnet2calc.models import get_model_path


class AIMNet2Calculator:
    """ Genegic AIMNet2 calculator 
    A helper class to load AIMNet2 models and perform inference.
    """

    keys_in = {
        'coord': torch.float,
        'numbers': torch.int,
        'charge': torch.float
        }
    keys_in_optional = {
        'mult': torch.float,
        'mol_idx': torch.int,
        'nbmat': torch.int,
        'nbmat_lr': torch.int,
        'nb_pad_mask': torch.bool,
        'nb_pad_mask_lr': torch.bool,
        'shifts': torch.float,
        'shifts_lr': torch.float,
        'cell': torch.float
        }
    keys_out = ['energy', 'charges', 'forces', 'hessian', 'stress']
    atom_feature_keys = ['coord', 'numbers', 'charges', 'forces']
    
    def __init__(self, model: Union[str, torch.nn.Module] = 'aimnet2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if isinstance(model, str):
            p = get_model_path(model)
            self.model = torch.jit.load(p, map_location=self.device)
        elif isinstance(model, nn.Module):
            self.model = model.to(self.device)
        else:
            raise AttributeError('Invalid model type/name.')

        self.cutoff = self.model.cutoff
        self.lr = hasattr(self.model, 'cutoff_lr')
        self.cutoff_lr = getattr(self.model, 'cutoff_lr', float('inf'))

        # indicator if input was flattened
        self._batch = None
        # placeholder for tensors that require grad
        self._saved_for_grad = None
        # set flag of current Coulomb model
        coul_methods = set(getattr(mod, 'method', None) for mod in iter_lrcoulomb_mods(self.model))
        assert len(coul_methods) <= 1, 'Multiple Coulomb modules found.'
        if len(coul_methods):
            self._coulomb_method = coul_methods.pop()
        else:
            self._coulomb_method = None

    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    def set_lrcoulomb_method(self, method, cutoff=15.0, dsf_alpha=0.2):
        assert method in ('simple', 'dsf', 'ewald'), f'Invalid method: {method}'
        if method == 'simple':
            for mod in iter_lrcoulomb_mods(self.model):
                mod.method = 'simple'
                self.cutoff_lr = float('inf')
        elif method == 'dsf':
            for mod in iter_lrcoulomb_mods(self.model):
                mod.method = 'dsf'
                self.cutoff_lr = cutoff
                mod.dsf_alpha = dsf_alpha
        elif method == 'ewald':
            for mod in iter_lrcoulomb_mods(self.model):
                mod.method = 'ewald'
                self.cutoff_lr = cutoff
        self._coulomb_method = method

    def eval(self, data: Dict[str, Any], forces=False, stress=False, hessian=False) -> Dict[str, Tensor]:
        data = self.prepare_input(data)
        if hessian and data['mol_idx'][-1] > 0:
            raise NotImplementedError('Hessian calculation is not supported for multiple molecules')
        data = self.set_grad_tensors(data, forces=forces, stress=stress, hessian=hessian)
        with torch.jit.optimized_execution(False):
            data = self.model(data)
        data = self.get_derivatives(data, forces=forces, stress=stress, hessian=hessian)
        data = self.process_output(data)
        return data
        
    def prepare_input(self, data: Dict[str, Any]) -> Dict[str, Tensor]:
        data = self.to_input_tensors(data)
        data = self.mol_flatten(data)
        if data.get('cell') is not None:
            if data['mol_idx'][-1] > 0:
                raise NotImplementedError('PBC with multiple molecules is not implemented yet.')
            if self._coulomb_method == 'simple':
                print('Switching to DSF Coulomb for PBC')
                self.set_lrcoulomb_method('dsf')
        data = self.make_nbmat(data)
        data = self.pad_input(data)
        return data
    
    def process_output(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data = self.unpad_output(data)
        data = self.mol_unflatten(data)
        data = self.keep_only(data)
        return data

    def to_input_tensors(self, data: Dict[str, Any]) -> Dict[str, Tensor]:
        ret = dict()
        for k in self.keys_in:
            assert k in data, f'Missing key {k} in the input data'
            # always detach !!
            ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in[k]).detach()
        for k in self.keys_in_optional:
            if k in data and data[k] is not None:
                ret[k] = torch.as_tensor(data[k], device=self.device, dtype=self.keys_in_optional[k]).detach()
        # convert any scalar tensors to shape (1,) tensors
        for k, v in ret.items():
            if v.ndim == 0:
                ret[k] = v.unsqueeze(0)
        return ret

    def mol_flatten(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        assert data['coord'].ndim in {2, 3}, 'Expected 2D or 3D tensor for coord'
        if data['coord'].ndim == 3:
            B, N = data['coord'].shape[:2]
            self._batch = B
            data['mol_idx'] = torch.repeat_interleave(torch.arange(0, B, device=self.device), torch.full((B,), N, device=self.device))
            for k, v in data.items():
                if k in self.atom_feature_keys:
                    assert v.ndim >= 2, f'Expected at least 2D tensor for {k}, got {v.ndim}D'
                    data[k] = v.flatten(0, 1)
        else:
            self._batch = None
            if 'mol_idx' not in data:
                data['mol_idx'] = torch.zeros(data['coord'].shape[0], device=self.device)
        return data
    
    def mol_unflatten(self, data: Dict[str, Tensor], batch=None) -> Dict[str, Tensor]:
        batch = batch or self._batch
        if batch is not None:
            for k, v in data.items():
                if k in self.atom_feature_keys:
                    data[k] = v.view(self._batch, -1, *v.shape[1:])
        return data
    
    def make_nbmat(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if 'cell' in data and data['cell'] is not None:
            assert data['cell'].ndim == 2, 'Expected 2D tensor for cell'
            if 'nbmat' not in data:
                data['coord'] = move_coord_to_cell(data['coord'], data['cell'])
                mat_idxj, mat_pad, mat_S = nblists_torch_pbc(data['coord'], data['cell'], self.cutoff)
                data['nbmat'], data['nb_pad_mask'], data['shifts'] = mat_idxj, mat_pad, mat_S
                if self.lr:
                    if 'nbmat_lr' not in data:
                        assert self.cutoff_lr < torch.inf, 'Long-range cutoff must be finite for PBC'
                        data['nbmat_lr'], data['nb_pad_mask_lr'], data['shifts_lr'] = nblists_torch_pbc(data['coord'], data['cell'], self.cutoff_lr)
                        data['cutoff_lr'] = torch.tensor(self.cutoff_lr, device=self.device)
        else:
            if 'nbmat' not in data:
                data['nbmat'] = nblist_torch_cluster(data['coord'], self.cutoff, data['mol_idx'], max_nb=128)
                if self.lr:
                    if 'nbmat_lr' not in data:
                        data['nbmat_lr'] = nblist_torch_cluster(data['coord'], self.cutoff_lr, data['mol_idx'], max_nb=1024)
                    data['cutoff_lr'] = torch.tensor(self.cutoff_lr, device=self.device)
        return data
    
    def pad_input(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        N = data['nbmat'].shape[0]
        data['coord'] = maybe_pad_dim0(data['coord'], N)
        data['numbers'] = maybe_pad_dim0(data['numbers'], N)
        data['mol_idx'] = maybe_pad_dim0(data['mol_idx'], N, value=data['mol_idx'][-1])
        return data

    def unpad_output(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        N = data['nbmat'].shape[0] - 1
        for k, v in data.items():
            if k in self.atom_feature_keys:
                data[k] = maybe_unpad_dim0(v, N)
        return data
    
    def set_grad_tensors(self, data: Dict[str, Tensor], forces=False, stress=False, hessian=False) -> Dict[str, Tensor]:
        self._saved_for_grad = dict() 
        if forces or hessian:
            data['coord'].requires_grad_(True)
            self._saved_for_grad['coord'] = data['coord']
        if stress:
            assert 'cell' in data, 'Stress calculation requires cell'
            scaling = torch.eye(3, requires_grad=True, dtype=data['cell'].dtype, device=data['cell'].device)
            data['coord'] = data['coord'] @ scaling
            data['cell'] = data['cell'] @ scaling
            self._saved_for_grad['scaling'] = scaling
        return data
    
    def keep_only(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ret = dict()
        for k, v in data.items():
            if k in self.keys_out or (k.endswith('_std') and k[:-4] in self.keys_out):
                ret[k] = v
        return ret
    
    def get_derivatives(self, data: Dict[str, Tensor], forces=False, stress=False, hessian=False) -> Dict[str, Tensor]:
        training = getattr(self.model, 'training', False)
        _create_graph = hessian or training
        x = []
        if hessian:
            forces = True
        if forces and ('forces' not in data or (_create_graph and not data['forces'].requires_grad)):
            forces = True
            x.append(self._saved_for_grad['coord'])
        if stress:
            x.append(self._saved_for_grad['scaling'])
        if x:
            tot_energy = data['energy'].sum()
            deriv = torch.autograd.grad(tot_energy, x, create_graph=_create_graph)
            if forces:
                data['forces'] = - deriv[0]
            if stress:
                if not forces:
                    dedc = deriv[0]
                else:
                    dedc = deriv[1]
                data['stress'] = dedc / data['cell'].detach().det().abs()
        if hessian:
            data['hessian'] = self.calculate_hessian(data['forces'], self._saved_for_grad['coord'])
        return data

    @staticmethod    
    def calculate_hessian(forces, coord):
        # here forces have shape (N, 3) and coord has shape (N+1, 3)
        # return hessian with shape (N, 3, N, 3)
        hessian = - torch.stack([
            torch.autograd.grad(_f, coord, retain_graph=True)[0]
            for _f in forces.flatten().unbind()
        ]).view(-1, 3, coord.shape[0], 3)[:-1, :, :-1, :]
        return hessian

    
def maybe_pad_dim0(a: Tensor, N: int, value=0.0) -> Tensor:
    _shape_diff = N - a.shape[0]
    assert _shape_diff == 0 or _shape_diff == 1, 'Invalid shape'
    if _shape_diff == 1:
        a = pad_dim0(a, value=value)
    return a

def pad_dim0(a: Tensor, value=0.0) -> Tensor:
    shapes = [0] * ((a.ndim - 1)*2) + [0, 1]
    a = torch.nn.functional.pad(a, shapes, mode='constant', value=value)
    return a

def maybe_unpad_dim0(a: Tensor, N: int) -> Tensor:
    _shape_diff = a.shape[0] - N
    assert _shape_diff == 0 or _shape_diff == 1, 'Invalid shape'
    if _shape_diff == 1:
        a = a[:-1]
    return a

def move_coord_to_cell(coord, cell):
    coord_f = coord @ cell.inverse()
    coord_f = coord_f % 1
    return coord_f @ cell


def _named_children_rec(module):
    if isinstance(module, torch.nn.Module):
        for name, module in module.named_children():
            yield name, module
            yield from _named_children_rec(module)


def iter_lrcoulomb_mods(model):
    for name, module in _named_children_rec(model):
        if name == 'lrcoulomb':
            yield module

