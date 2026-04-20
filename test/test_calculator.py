import os

import ase.io
import numpy as np

from aimnet2calc.calculator import AIMNet2Calculator

MODELS = ('aimnet2', 'aimnet2_b973c')
DIR = os.path.dirname(__file__)


def _struct_pbc():
    filename = os.path.join(DIR, '1008775.cif')
    atoms = ase.io.read(filename)
    ret = dict()
    ret['coord'] = atoms.positions
    ret['numbers'] = atoms.numbers
    ret['charge'] = 0.0
    ret['cell'] = atoms.cell.array
    return ret


def _struct_list():
    filename = os.path.join(DIR, 'mols_size_var.xyz')
    atoms = ase.io.read(filename, index=':')
    ret = dict()
    ret['coord'] = np.concatenate([a.positions for a in atoms])
    ret['numbers'] = np.concatenate([a.numbers for a in atoms])
    ret['mol_idx'] = np.concatenate([[i] * len(a) for i, a in enumerate(atoms)])
    ret['charge'] = [0.0] * len(atoms)
    return ret


def _stuct_batch():
    filename = os.path.join(DIR, 'mols_size_36.xyz')
    atoms = ase.io.read(filename, index=':')
    ret = dict()
    ret['coord'] = [a.positions for a in atoms]
    ret['numbers'] = [a.numbers for a in atoms]
    ret['charge'] = [0.0] * len(atoms)
    return ret


def _test_energy(calc, data):
    _out = calc(data)
    assert 'energy' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == False


def _test_forces(calc, data):
    _out = calc(data, forces=True)
    assert 'energy' in _out
    assert 'forces' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == True
    assert len(_out['forces']) == len(data['coord']), _out['forces'].shape
    assert _out['forces'].requires_grad == False


def _test_forces_stress(calc, data):
    _out = calc(data, forces=True, stress=True)
    assert 'energy' in _out
    assert 'forces' in _out
    assert 'stress' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == True
    assert len(_out['forces']) == len(data['coord'])
    assert _out['forces'].requires_grad == False
    assert len(_out['stress']) == 3
    assert _out['stress'].requires_grad == False


def _test_hessian(calc, data):
    _out = calc(data, hessian=True)
    assert 'energy' in _out
    assert 'forces' in _out
    assert 'hessian' in _out
    assert len(_out['energy']) == len(data['charge'])
    assert _out['energy'].requires_grad == True
    assert len(_out['forces']) == len(data['coord'])
    assert _out['forces'].requires_grad == False
    assert len(_out['hessian']) == len(data['coord'])
    assert _out['hessian'].requires_grad == False


def test_calculator():
    for model in MODELS:
        print('Testing model:', model)
        calc = AIMNet2Calculator(model)
        for data, typ in zip((_stuct_batch(), _struct_list(), _struct_pbc()), ('batch', 'list', 'pbc')):
            if typ == 'pbc' and not (calc.cutoff_lr < float('inf')):
                print('Skipping PBC with LR')
                continue
            print('Testing data:', typ)
            print('energy: ', _test_energy(calc, data))
            print('forces: ', _test_forces(calc, data))
            if len(data['charge']) == 1:
                print('hessian: ', _test_hessian(calc, data))
            if typ == 'pbc':
                print('forces+stress: ', _test_forces_stress(calc, data))


if __name__ == '__main__':
    test_calculator()