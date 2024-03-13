import torch
from torch import nn, Tensor
from typing import Dict, List


class EnsembledModel(nn.Module):
    """Create ensemble of AIMNet2 models."""

    def __init__(
        self,
        models: List,
        x=["coord", "numbers", "charge"],
        out=["energy", "forces", "charges"],
        detach=True,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.x = x
        self.out = out
        self.detach = detach

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res: List[Dict[str, Tensor]] = []
        for model in self.models:
            _in = dict()
            for k in data:
                if k in self.x:
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
            data[k + "_std"] = vv.std(dim=0)

        return data
