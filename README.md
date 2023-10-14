
# AIMNet2: a general-purpose neural netrork potential for organic and element-organic molecules.

The repository contains AIMNet2 models, example Python code and supplementary data for the manuscript

**AIMNet2: A Neural Network Potential to Meet your Neutral, Charged, Organic, and Elemental-Organic Needs**
*Dylan Anstine ,Roman Zubatyuk ,Olexandr Isayev*
[10.26434/chemrxiv-2023-296ch](https://doi.org/10.26434/chemrxiv-2023-296ch)
  
## Models

The models are applicable for systems containing the following set of chemical elements:
{H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I}, both neutral and charged. The models aim to reproduce RKS B97-3c and wB97M-D3 energies.
  
The models are in the form of JIT-compiled PyTorch-2.0 files and could be loaded in Python or C++ code.

Note, that at present models have O(N^2) compute and memory complexity w.r.t. number of atoms. They could be applied to systems up to a few 100's of atoms. Linear scaling models, with the same parametrization, will be released soon.

In Python, the models could be loaded with the `torch.jit.load` function. As an input, they accept single argument of type `Dict[str, Tensor]` with the following data:
```
coords: shape (m, n, 3) - atomic coordinates in Angstrom 
numbers: shape (m, n) - atomic numbers
charge: shape (m, ) - total charge
```
Output is a dictionary with the following keys:
```
energy: shape (m, ) - energy in eV
charges: shape (m, n) - partial atomic charges
```
## Calculators

We provide example code for AIMNet2 calculators for [ASE](https://wiki.fysik.dtu.dk/ase) and [pysisyphus](https://pysisyphus.readthedocs.io/) Python libraries. The code shows an example use of the AIMNet2 models. 

We also provide example geometry optimization scripts with ASE and Pysisyphus, and `pysis_mod` script which is a drop-in replacement for Pysisyphus `pysis` command-line utility, with AIMNet2 calculator enabled.


### Feedback

We would appreciate it if you could share feedback about model accuracy and performance. This would be important not only for us, to guide further developments of the model, but for the broad community as well. 
Please share your thoughts and experience, either positive or negative, by opening an issue in this repo.

