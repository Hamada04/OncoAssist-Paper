"""Google Colab command-line release interface for controlled Primary V1 studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from oncoassist_research import controlled_runner


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mge", required=True, type=Path, help="Explicit mGE CSV path.")
    parser.add_argument("--mdm", required=True, type=Path, help="Explicit mDM CSV path.")
    parser.add_argument("--mcna", required=True, type=Path, help="Explicit mCNA CSV path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Persistent controlled-study output root.")
    parser.add_argument("--run-id", required=True, help="Explicit prospective run identifier.")
    parser.add_argument("--root-seed", required=True, type=int, help="Explicit non-negative root seed.")
    parser.add_argument("--ae-device", required=True, choices=("cpu", "gpu"), help="Explicit TensorFlow AE device policy.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled Primary V1 Colab release entrypoint.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "status", "resume"):
        _add_common_arguments(commands.add_parser(name))
    recovery = commands.add_parser("recover-abandoned-lock")
    _add_common_arguments(recovery)
    recovery.add_argument("--expected-lock-id", required=True, help="Exact abandoned lock ID confirmed by the operator.")
    return parser


def _config(arguments: argparse.Namespace) -> controlled_runner.ControlledRunnerConfig:
    missing = [(name, path) for name, path in (("mGE", arguments.mge), ("mDM", arguments.mdm), ("mCNA", arguments.mcna)) if not path.is_file()]
    if missing:
        names = ", ".join(name for name, _ in missing)
        raise FileNotFoundError(f"Required canonical source CSV path(s) do not exist: {names}.")
    return controlled_runner.ControlledRunnerConfig(
        arguments.mge,
        arguments.mdm,
        arguments.mcna,
        arguments.output_dir,
        arguments.run_id,
        arguments.root_seed,
        arguments.ae_device,
    )


def _status(prepared: controlled_runner.PreparedStudy) -> dict[str, Any]:
    directory = prepared.study_directory
    if directory.is_dir():
        state = controlled_runner.reconstruct_runtime_state(directory, prepared.binding)
        coordinates = state["coordinates"]
    else:
        state = {"derived_state": "NEW", "complete_coordinate_count": 0}
        coordinates = []
    evaluation_only = sum(item.get("resume_classification") == "EVALUATION_ONLY_RESUME" for item in coordinates)
    complete = sum(item.get("resume_classification") == "COMPLETE" for item in coordinates)
    failures = len(list((directory / "failures").glob("*.json"))) if directory.is_dir() else 0
    return {
        "study_identity_sha256": prepared.binding.study_identity_sha256,
        "dataset_provenance_identity_sha256": prepared.provenance.identity_sha256,
        "protocol_identity_sha256": prepared.protocol.identity_sha256,
        "immutable_reference_sha256": prepared.binding.payload["immutable_reference_sha256"],
        "feature_provenance_status": prepared.protocol.feature_provenance_status,
        "ae_device_policy": prepared.config.ae_device_policy,
        "study_directory": str(directory),
        "derived_state": state["derived_state"],
        "coordinate_progress": {
            "total": 25,
            "completed": complete,
            "evaluation_only_resume": evaluation_only,
            "remaining": 25 - complete,
            "failure_record_count": failures,
        },
        "current_coordinate": None,
    }


def _prepare(arguments: argparse.Namespace, *, initialize: bool) -> controlled_runner.PreparedStudy:
    prepared = controlled_runner.prepare_study(_config(arguments))
    if initialize:
        controlled_runner.initialize_study(prepared)
    elif prepared.study_directory.exists():
        controlled_runner.validate_study_directory(prepared.study_directory, prepared.binding)
    return prepared


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "status":
            print(json.dumps(_status(_prepare(arguments, initialize=False)), indent=2, sort_keys=True))
            return 0
        prepared = _prepare(arguments, initialize=True)
        if arguments.command == "preflight":
            print(json.dumps({"status": _status(prepared), "preflight": prepared.preflight}, indent=2, sort_keys=True))
            return 0
        if arguments.command == "recover-abandoned-lock":
            controlled_runner.recover_abandoned_study_lock(prepared.study_directory, prepared.binding, arguments.expected_lock_id)
            print(json.dumps({"status": "abandoned_lock_recovered", **_status(prepared)}, indent=2, sort_keys=True))
            return 0
        results = controlled_runner.run_study(prepared)
        print(json.dumps({"status": _status(prepared), "coordinate_results": [result.__dict__ for result in results]}, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"controlled-runner error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
