import pytest
from pyaimnet2 import load_model


@pytest.mark.parametrize(
    "model",
    [pytest.param("b973c", id="b973c"), pytest.param("wb97m-d3", id="wb97m-d3")],
)
def test_loading_models(model):
    """test loading different models using the package."""

    _ = load_model(model_name=model)


def test_missing_models():
    """Make sure an error is raised if we try and load a model not supported"""

    with pytest.raises(FileNotFoundError):
        _ = load_model(model_name="b3lyp")


@pytest.mark.parametrize(
    "model, expected_energy",
    [
        pytest.param("wb97m-d3", -2081.0415, id="wb97m-d3"),
        pytest.param("b973c", -2078.9055, id="b973c"),
    ],
)
def test_model_energies(model, expected_energy, water):
    """Test the energies calculated for a water molecule match the expected values."""

    aimnet2 = load_model(model_name=model)
    return_data = aimnet2(water)
    assert return_data["energy"] == pytest.approx(expected_energy)
