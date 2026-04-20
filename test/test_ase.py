import os

import ase
import numpy as np

from aimnet2calc.aimnet2ase import AIMNet2ASE

MODELS = ('aimnet2', 'aimnet2_b973c')
DIR = os.path.dirname(__file__)


def _struct_pbc():
    filename = os.path.join(DIR, '1008775.cif')
    return ase.io.read(filename)


def _struct_list():
    filename = os.path.join(DIR, 'mols_size_var.xyz')
    return ase.io.read(filename, index=':')


def _stuct_batch():
    filename = os.path.join(DIR, 'mols_size_36.xyz')
    return ase.io.read(filename, index=':')


def _test_dipole(calc, atoms):
    atoms.calc = calc
    e =atoms.get_potential_energy()
    assert isinstance(e, float)

    assert hasattr(atoms, 'get_charges')
    q = atoms.get_charges()
    assert q.shape == (len(atoms),)
    
    assert hasattr(atoms, 'get_dipole_moment')
    dm = atoms.get_dipole_moment()
    assert dm.shape == (3,)
    

def test_calculator():
    for model in MODELS:
        print('Testing model:', model)
        calculator = AIMNet2ASE(model)
        for atoms_list, runtype in zip((_stuct_batch(), _struct_list(), _struct_pbc()), ('batch', 'list', 'pbc')):
            if runtype == 'batch' or runtype == 'list':
                for atoms in atoms_list:
                    _test_dipole(calculator, atoms)

            else:
                _test_dipole(calculator, atoms_list)


