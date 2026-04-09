import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _ensure_src_namespace() -> None:
    """Create a minimal src package namespace when imported as top-level dataset."""
    if "src" in sys.modules:
        return

    src_module = ModuleType("src")
    src_module.__path__ = [str(Path(__file__).resolve().parents[1])]
    src_module.__package__ = "src"
    sys.modules["src"] = src_module


def _load_dataset_submodule(module_name: str):
    """Load dataset submodule under src.dataset.* to keep relative imports valid."""
    full_name = f"src.dataset.{module_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    module_path = Path(__file__).resolve().parent / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {full_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "dataset":
    _ensure_src_namespace()
    # Alias this package so submodules resolve as src.dataset.*.
    sys.modules.setdefault("src.dataset", sys.modules[__name__])

    AnnotationDataset = _load_dataset_submodule("AnnotationDataset").AnnotationDataset
    Vocabulary = _load_dataset_submodule("vocabulary").Vocabulary
else:
    from .AnnotationDataset import AnnotationDataset
    from .vocabulary import Vocabulary

__all__ = ["AnnotationDataset", "Vocabulary"]
