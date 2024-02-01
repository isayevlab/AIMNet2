from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.elem_data import ATOMIC_NUMBERS
from pysisyphus.constants import BOHR2ANG, ANG2BOHR, AU2EV
import torch
import os


EV2AU = 1 / AU2EV


class AIMNet2Calculator(Calculator):
    def __init__(self, model=None, charge=0, **kwargs):
        super().__init__(charge=charge, **kwargs)
        if model is None:
            model = os.environ.get('AIMNET_MODEL')
        if isinstance(model, str):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = torch.jit.load(model, map_location=device)
        assert isinstance(model, torch.nn.Module)
        self.model = model
        self.charge = charge
        self.device = next(self.model.parameters()).device

    def _prepere_input(self, atoms, coord):
        numbers = torch.as_tensor([[ATOMIC_NUMBERS[a.lower()] for a in atoms]], device=self.device)
        coord = torch.as_tensor(coord, dtype=torch.float, device=self.device).view(1, numbers.shape[1], 3) * BOHR2ANG
        charge = torch.as_tensor([self.charge], dtype=torch.float, device=self.device)
        return dict(coord=coord, numbers=numbers, charge=charge)

    def get_energy(self, atoms, coords):
        _in = self._prepere_input(atoms, coords)
        with torch.no_grad(), torch.jit.optimized_execution(False):
            _out = self.model(_in)
        energy = _out['energy'].item() * EV2AU
        return dict(energy=energy)

    def get_forces(self, atoms, coords):
        _in = self._prepere_input(atoms, coords)
        with torch.jit.optimized_execution(False):
            _in['coord'].requires_grad_(True)
            _out = self.model(_in)
            e = _out['energy']
            f = - torch.autograd.grad(e, _in['coord'])[0]
        energy = e.item() * EV2AU
        forces = (f * (EV2AU / ANG2BOHR))[0].flatten().cpu().numpy()
        return dict(energy=energy, forces=forces)
    
    def get_hessian(self, atoms, coords):
        _in = self._prepere_input(atoms, coords)
        with torch.jit.optimized_execution(False):
            _in['coord'].requires_grad_(True)
            _out = self.model(_in)
            e = _out['energy']
            f = -_get_derivatives_not_none(_in['coord'], e, create_graph=True)
            h = - torch.stack([
            _get_derivatives_not_none(_in['coord'], _f, retain_graph=True)[0]
                    for _f in f.flatten().unbind()
                    ])
        energy = e.item() * EV2AU
        forces = (f.detach() * (EV2AU / ANG2BOHR))[0].flatten().to(torch.double).cpu().numpy()
        hessian = (h.flatten(-2, -1) * (EV2AU / ANG2BOHR / ANG2BOHR)).to(torch.double).cpu().numpy()
        return dict(energy=energy, forces=forces, hessian=hessian)
        

def _get_derivatives_not_none(x, y, retain_graph=None, create_graph=False):
    ret = torch.autograd.grad([y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph)[0]
    assert ret is not None
    return ret






