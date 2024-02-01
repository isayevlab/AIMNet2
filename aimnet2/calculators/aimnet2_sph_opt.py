import torch
from openbabel import pybel
import numpy as np
import re
import os
from aimnet2sph import AIMNet2Calculator
from pysisyphus.Geometry import Geometry
from pysisyphus.elem_data import INV_ATOMIC_NUMBERS
from pysisyphus.constants import ANG2BOHR
from pysisyphus.helpers import do_final_hessian


def pybel2geom(mol, coord_type='dlc'):
    coord = np.array([a.coords for a in mol.atoms]).flatten() * ANG2BOHR
    atoms = [INV_ATOMIC_NUMBERS[a.atomicnum].lower() for a in mol.atoms]
    geom = Geometry(atoms, coord, coord_type=coord_type)
    return geom


def update_mol(mol, geom, align=True):
    # make copy
    mol_old = pybel.Molecule(pybel.ob.OBMol(mol.OBMol))
    # update coord
    coord = geom.coords3d / ANG2BOHR
    for i, c in enumerate(coord):
        mol.OBMol.GetAtom(i+1).SetVector(*c.tolist())
    # align
    if align:
        aligner = pybel.ob.OBAlign(False, False)
        aligner.SetRefMol(mol_old.OBMol)
        aligner.SetTargetMol(mol.OBMol)
        aligner.Align()
        rmsd = aligner.GetRMSD()
        aligner.UpdateCoords(mol.OBMol)
        print(f'RMSD: {rmsd:.2f} Angs')


def guess_pybel_type(filename):
    assert '.' in filename
    return os.path.splitext(filename)[1][1:]


def guess_charge(mol):
    m = re.search('charge: (-?\d+)', mol.title)
    if m:
        charge = int(m.group(1))
    else:
        charge = mol.charge
    return charge


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--charge', type=int, default=None, help='Molecular charge (default: check molecule title for "charge: {int}" or get total charge from OpenBabel).')
    parser.add_argument('--ts', action='store_true', help='Do TS optimization')
    parser.add_argument('--coord', type=str, help='Coordinates type, e.g. cart, dlc (default) or redund')
    parser.add_argument('--thresh', type=str, default='gau_loose', help='Optimization threshold, one of aaug_loose (default), gau, gau_tight, gau_vtight, baker.')
    parser.add_argument('model')
    parser.add_argument('in_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading AIMNet2 model from file', args.model)
    model = torch.jit.load(args.model, map_location=device)
    calc = AIMNet2Calculator(model)

    in_format = guess_pybel_type(args.in_file)
    out_format = guess_pybel_type(args.out_file)

    if args.ts:
        from pysisyphus.tsoptimizers import RSPRFOptimizer as Opt
        opt_kwargs = dict(assert_neg_eigval=True, hessian_recalc=30)
    else:
        from pysisyphus.optimizers.RFOptimizer import RFOptimizer as Opt
        opt_kwargs = dict()

    with open(args.out_file, 'w') as f:
        for mol in pybel.readfile(in_format, args.in_file):
            geom = pybel2geom(mol)
            charge = args.charge if args.charge is not None else guess_charge(mol)
            calc.charge = charge
            geom.set_calculator(calc)
            opt = Opt(geom, thresh=args.thresh, max_cycles=1000, **opt_kwargs)
            
            with torch.jit.optimized_execution(False):
                opt.run()

            if args.ts:
                do_final_hessian(geom, save_hessian=False, is_ts=args.ts, print_thermo=True)

            update_mol(mol, geom, align=False)
            f.write(mol.write(out_format))
            f.flush()
