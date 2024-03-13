import pytest
import torch


@pytest.fixture(scope="package")
def water() -> dict:
    """Return a dict of a water molecule which can be used with the aimnet2 model to check the energy"""
    return {
        "coord": torch.tensor(
            [
                [
                    [0.06112021, 0.38865671, 0.05890042],
                    [0.72378355, -0.31162935, -0.03823509],
                    [-0.78490371, -0.07702733, -0.02066534],
                ],
            ],
            dtype=torch.float64,
        ),
        "numbers": torch.tensor([[8, 1, 1]], dtype=torch.long),
        "charge": torch.tensor([0], dtype=torch.float64),
    }
