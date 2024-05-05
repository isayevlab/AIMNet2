from setuptools import setup, find_packages

setup(
    name='aimnet2calc',
    version='0.0.1',
    author='Roman Zubatyuk',
    author_email='zubatyuk@gmail.com',
    description='Interface for AIMNet2 models',
    packages=find_packages(),
    install_requires=[
        'torch>2.0,<3',
        'torch-cluster',
        'numpy',
        'numba',
        'ase',
        'pysisyphus',
        'requests',
        # 'openbabel'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'aimnet2pysis=aimnet2calc.aimnet2pysis:run_pysis'
        ],
    },
)