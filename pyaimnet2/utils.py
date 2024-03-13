from typing import Literal

available_models = ["b973c", "wb97m-d3"]
MODELS = Literal["b973c", "wb97m-d3"]


def load_model(model_name: MODELS):
    """
    Load the specified AIMNET2 model.

    Args:
        model_name: The name of the ensemble model which should be loaded (`b973c`, `wb97m-d3`).

    Returns:
        A aimnet2 model ready for evaluation.
    """
    from importlib.resources import files
    import pathlib
    import torch.jit

    if model_name.lower() not in available_models:
        raise FileNotFoundError(
            f"The model {model_name} is not available chose from {available_models}"
        )

    package_path: pathlib.Path = files("pyaimnet2")
    model_path = package_path.joinpath("models", f"aimnet2_{model_name}_ens.jpt")
    return torch.jit.load(model_path.as_posix())
