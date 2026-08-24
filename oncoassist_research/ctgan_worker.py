"""Standalone CPU-only minority CTGAN worker. Do not import from the parent process."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys

# Running this file directly puts its directory first on sys.path. Remove that
# entry so ``import ctgan`` resolves the installed dependency, not the parent module.
_WORKER_DIRECTORY = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != _WORKER_DIRECTORY]

import numpy as np
import pandas as pd
import torch
import sdv
import ctgan
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


SCHEMA_VERSION = "research-minority-ctgan-worker-v1"


def _canonical(value):
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _array_hash(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(_canonical({"dtype": str(array.dtype), "shape": list(array.shape)}))
    digest.update(array.tobytes())
    return digest.hexdigest()


def preflight(response_path):
    """Report dependency/API compatibility without constructing, fitting, or sampling CTGAN."""
    parameters = list(inspect.signature(CTGANSynthesizer).parameters)
    required = {"metadata", "epochs", "batch_size", "pac", "verbose"}
    if required.difference(parameters):
        raise RuntimeError("CTGAN constructor API is incompatible; no fallback exists.")
    response = {
        "schema_version": SCHEMA_VERSION,
        "preflight_only": True,
        "ctgan_execution_backend": "isolated_cpu_subprocess_v1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ctgan_gpu_enabled": False,
        "constructor_parameters": parameters,
        "versions": {
            "sdv": sdv.__version__,
            "ctgan": getattr(ctgan, "__version__", "unknown"),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    Path(response_path).write_text(json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(request_path, input_path, output_path, response_path):
    tensorflow_present_before_ctgan = "tensorflow" in sys.modules
    request = json.loads(open(request_path, encoding="utf-8").read())
    with np.load(input_path) as payload:
        features = payload["features"]
    if request.get("strategy") != "minority_only_ctgan" or _array_hash(features) != request.get("input_minority_features_sha256"):
        raise ValueError("Minority CTGAN request/input contract is invalid.")
    names, config = request["feature_names"], request["ctgan_config"]
    if features.ndim != 2 or not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all() or len(names) != features.shape[1]:
        raise ValueError("Minority CTGAN features are invalid.")
    if hashlib.sha256(_canonical(names)).hexdigest() != request["feature_names_sha256"]:
        raise ValueError("Minority CTGAN feature schema hash is invalid.")
    parameters = list(inspect.signature(CTGANSynthesizer).parameters)
    required = {"metadata", "epochs", "batch_size", "pac", "verbose"}
    if required.difference(parameters):
        raise RuntimeError("CTGAN constructor API is incompatible; no fallback exists.")
    seed = int(request["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    training = pd.DataFrame(features, columns=names)
    metadata = SingleTableMetadata(); metadata.detect_from_dataframe(data=training)
    if callable(getattr(metadata, "validate", None)):
        metadata.validate()
    kwargs = {"metadata": metadata, **config}
    if "enable_gpu" in parameters:
        kwargs["enable_gpu"] = False
    synthesizer = CTGANSynthesizer(**kwargs)
    synthesizer.fit(training)
    requested = int(request["requested_synthetic_rows"])
    samples = synthesizer.sample(requested)
    if samples.columns.tolist() != names:
        raise ValueError("Minority CTGAN synthetic feature schema is invalid.")
    synthetic = np.ascontiguousarray(samples.to_numpy(dtype=np.float32, copy=True))
    if synthetic.shape != (requested, len(names)) or not np.isfinite(synthetic).all():
        raise ValueError("Minority CTGAN synthetic output is invalid.")
    losses = {"available": False}
    getter = getattr(synthesizer, "get_loss_values", None)
    if callable(getter):
        values = getter()
        losses = {"available": True, "row_count": int(len(values))} if isinstance(values, pd.DataFrame) else {"available": True}
    np.savez_compressed(output_path, synthetic=synthetic)
    response = {
        "schema_version": SCHEMA_VERSION, "strategy": "minority_only_ctgan", "feature_names": names,
        "feature_names_sha256": request["feature_names_sha256"], "requested_synthetic_rows": requested,
        "returned_synthetic_rows": len(synthetic), "input_minority_features_sha256": _array_hash(features),
        "synthetic_sha256": _array_hash(synthetic), "synthetic_dtype": str(synthetic.dtype),
        "constructor_configuration": {"metadata_supplied": True, **{key: value for key, value in kwargs.items() if key != "metadata"}},
        "versions": {"sdv": sdv.__version__, "ctgan": getattr(ctgan, "__version__", "unknown"), "torch": str(torch.__version__), "numpy": np.__version__, "pandas": pd.__version__},
        "seed_evidence": {"random_seed_requested": seed, "python_seed": seed, "numpy_seed": seed, "pytorch_seed": seed, "ctgan_constructor_seed_control": "unavailable_in_public_constructor", "exact_regeneration_guaranteed": False},
        "metadata_schema": {"latent_columns_only": True, "controlled_outcome_column": None}, "loss_summary": losses,
        "execution_evidence": {"ctgan_execution_backend": "isolated_cpu_subprocess_v1", "tensorflow_present_in_worker": tensorflow_present_before_ctgan, "ctgan_gpu_enabled": False},
    }
    open(response_path, "w", encoding="utf-8").write(json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--preflight":
        preflight(sys.argv[2])
    else:
        main(*sys.argv[1:])
