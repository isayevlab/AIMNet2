**__ Update 6/10/24 __**
We release new code, suaitable for large molecules and perioric calculations. Old code available in the **old** branch. Models were re-compiled and are not compatible with the new code. 


# AIMNet2 Calculator: Fast, Accurate Molecular Simulations

This package integrates the powerful AIMNet2 neural network potential into your simulation workflows. AIMNet2 provides fast and reliable energy, force, and property calculations for molecules containing a diverse range of elements.

## Key Features:

- **Accurate and Versatile:** AIMNet2 excels at modeling neutral, charged, organic, and elemental-organic systems.
- **Flexible Interfaces:** Use AIMNet2 through convenient calculators for popular simulation packages like ASE and PySisyphus.
- **Flexible Long-Range Interactions:** Optionally employ the Dumped-Shifted Force (DSF) or Ewald summation Coulomb models for accurate calculations in large or periodic systems.


## Getting Started

### 1. Installation

While package is in alpha stage and repository is private, please install into your conda envoronment manually with
```
# install requirements
conda install -y pytorch pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install -y -c pyg pytorch-cluster
conda install -y -c conda-forge openbabel ase
## pysis requirements
conda install -y -c conda-forge autograd dask distributed h5py fabric jinja2 joblib matplotlib numpy natsort psutil pyyaml rmsd scipy sympy scikit-learn
# now should not do any pip installs
pip install git+https://github.com/eljost/pysisyphus.git
# finally, this repo
git clone git@github.com:zubatyuk/aimnet2calc.git
cd aimnet2calc
python setup.py install
```

### 2. Available interfaces

#### ASE [[https://wiki.fysik.dtu.dk/ase]](https://wiki.fysik.dtu.dk/ase)

```
from aimnet2calc import AIMNet2ASE
calc = AIMNet2ASE('aimnet2')
```

To specify total molecular charge and spin multiplicity, use optional `charge` and `mult` keyword arguments, or  `set_charge` and `set_mult` methods:

```
calc = AIMNet2ASE('aimnet2', charge=1)
atoms1.calc = calc
# calculations on atoms1 will be done with charge 1
....
atoms2.calc = calc
calc.set_charge(-2)
# calculations on atoms1 will be done with charge -2
```

#### PySisyphus [[https://pysisyphus.readthedocs.io]](https://pysisyphus.readthedocs.io/)

```
from aimnet2calc import AIMNet2PySis
calc = AIMNet2PySis('aimnet2')
```

This produces standard PySisyphus calculator.

Instead of `Pysis` command line utility, use `aimnet2pysis`. This registeres AIMNet2 calculator with PySisyphus.
Example `calc` section for PySisyphus YAML files:

```
calc:
   type: aimnet              # use AIMNet2 calculator
   model: aimnet2_b973c      # use aimnet2_b973c_0.jpt model
```

### 3. Base calculator

```
from aimnet2calc import AIMNet2Calculator
```

#### Initialization

```
calc = AIMNet2Calculator('aimnet2')
```
will load default AIMNet2 model aimnet2_wb97m_0.jpt as defined at `aimnet2calc/models.py` . If file does not exist on the machine, it will be downloaded from [aimnet-model-zoo](http://github.com/zubatyuk/aimnet-model-zoo) repository.

```
calc = AIMNet2Calculator('/path/to_a/model.jpt')
```
will load model from the file. 

#### Input structure

The calculator accepts a dictionary containig lists, numpy arrays, torch tensors, or anything that could be accepted by `torch.as_tensor`. 

The input could be for a single molecule (dict keys and shapes):

```
coord: (B, N, 3)  # atomic coordinates in Angstrom
numbers (B, N)    # atomic numbers
charge (B,)       # molecular charge
mult (B,)         # spin multiplicity, optional
```

or for a concatenation of molecules:

```
coord: (N, 3)  # atomic coordinates in Angstrom
numbers (N,)    # atomic numbers
charge (B,)    # molecular charge
mult (B,)      # spin multiplicity, optional
mol_idx (N,)   # molecule index for each atom, should contain integers in increasing order, with (B-1) is the maximum number.
```

where `B` is the number of molecules, `N` is number of atoms. 


#### Calling calculator

```
results = calc(data, forces=False, stress=False, hessian=False)
```

`results` would be a dictionary of PyTorch tensors containing `energy`, `charges`, and possibly `forces`, `stress` and `hessian` if requested.

### 4. Long range Coulomb model

By default, Coulomb energy is calculated in O(N^2) manner, e.g. pair interaction between every pair of atoms in system. For very large or periodic systems, O(N) Dumped-Shifted Force Coulomb model could be employed [doi: 10.1063/1.2206581](https://doi.org/10.1063/1.2206581). With `AIMNet2Calculator` interface, switch between standard and DSF Coulomb implementations im AIMNet2 models:

```
# switch to O(N)
calc.set_lrcoulomb_method('dsf', cutoff=15.0, dsf_alpha=0.2)
# switch to O(N^2), not suitable for PBC
calc.set_lrcoulomb_method('simple')
```




