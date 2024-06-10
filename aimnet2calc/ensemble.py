import torch
from torch import Tensor, nn
from typing import List, Dict, Union



class Forces(nn.Module):
    """Compute forces from energy using autograd.
    """
    def __init__(self, module: nn.Module,
                 x: str = 'coord', y: str = 'energy', key_out: str = 'forces',
                 detach: bool = True):
        super().__init__()
        self.add_module('module', module)
        self.x = x
        self.y = y
        self.key_out = key_out
        self.detach = detach

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        y = data[self.y]
        create_graph = self.training or not self.detach
        g = torch.autograd.grad(
            [y.sum()], [data[self.x]], create_graph=create_graph)[0]
        assert g is not None
        data[self.key_out] = - g
        torch.set_grad_enabled(prev)
        return data


class EnsembledModel(nn.Module):
    def __init__(self, models: List[nn.Module],
                 out=['energy', 'forces', 'charges'],
                 detach=False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.out = out
        self.detach = detach

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res : List[Dict[str, Tensor]] = []
        for model in self.models:
            _in = dict()
            for k in data:
                _in[k] = data[k]
            _out = model(_in)
            _r = dict()
            for k in _out:
                if k in self.out:
                    _r[k] = _out[k]
                    if self.detach:
                        _r[k] = _r[k].detach()
            res.append(_r)

        for k in res[0]:
            v = []
            for x in res:
                v.append(x[k])
            vv = torch.stack(v, dim=0)
            data[k] = vv.mean(dim=0)
            data[k + '_std'] = vv.std(dim=0)

        return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--out-keys', type=str, nargs='+', default=['energy', 'forces', 'charges'])
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--forces', action='store_true')
    parser.add_argument('--grad-of-forces', action='store_true')
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    models = [torch.jit.load(m, map_location='cpu') for m in args.models]
    if args.forces:
        models = [Forces(m, detach=not args.grad_on_forces) for m in models]

    print('Ensembling {len(models)} models.')
    ens = EnsembledModel(models, out=args.out_keys, detach=args.detach)
    ens = torch.jit.script(ens)
    ens.save(args.output)

    

    