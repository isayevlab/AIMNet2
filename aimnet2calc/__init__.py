from .calculator import AIMNet2Calculator
__all__ = ['AIMNet2Calculator']

try:
    from .aimnet2ase import AIMNet2ASE
    __all__.append('AIMNet2ASE')
except ImportError:
    import warnings
    warnings.warn('ASE is not installed. AIMNet2ASE will not be available.')
    pass


