import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_primary_v1_colab as cli
from oncoassist_research import controlled_runner


class ColabReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.paths = {}
        for name in ("mge", "mdm", "mcna"):
            path = self.root / f"{name}.csv"
            path.write_text("SAMPLE_ID,CLASS,F\n", encoding="utf-8")
            self.paths[name] = path

    def tearDown(self):
        self.temporary.cleanup()

    def argv(self, command="preflight"):
        return [
            command,
            "--mge", str(self.paths["mge"]),
            "--mdm", str(self.paths["mdm"]),
            "--mcna", str(self.paths["mcna"]),
            "--output-dir", str(self.root / "outputs"),
            "--run-id", "release-test",
            "--root-seed", "17",
            "--ae-device", "cpu",
        ]

    def prepared(self):
        return SimpleNamespace(
            study_directory=self.root / "outputs" / ("a" * 64),
            binding=SimpleNamespace(study_identity_sha256="a" * 64, payload={"immutable_reference_sha256": "b" * 64}),
            provenance=SimpleNamespace(identity_sha256="c" * 64),
            protocol=SimpleNamespace(identity_sha256="d" * 64, feature_provenance_status="UNKNOWN"),
            config=SimpleNamespace(ae_device_policy="cpu"),
            preflight={"checked": True},
        )

    def test_cli_exposes_required_commands_and_no_scientific_override_arguments(self):
        parser = cli.build_parser()
        commands = next(action for action in parser._actions if action.dest == "command")
        self.assertTrue({"preflight", "run", "status", "resume"}.issubset(commands.choices))
        with self.assertRaises(SystemExit):
            cli.main([*self.argv(), "--candidate", "forbidden"])

    def test_missing_each_explicit_modality_path_fails_before_runner(self):
        for name in ("mge", "mdm", "mcna"):
            with self.subTest(modality=name):
                missing = self.paths[name]
                missing.unlink()
                with patch("run_primary_v1_colab.controlled_runner.prepare_study", side_effect=AssertionError):
                    self.assertEqual(cli.main(self.argv()), 2)
                missing.write_text("SAMPLE_ID,CLASS,F\n", encoding="utf-8")

    def test_preflight_initializes_runner_study_without_running_scientific_execution(self):
        prepared = self.prepared()
        with patch("run_primary_v1_colab.controlled_runner.prepare_study", return_value=prepared) as prepare, patch(
            "run_primary_v1_colab.controlled_runner.initialize_study"
        ) as initialize, patch("run_primary_v1_colab.controlled_runner.run_study", side_effect=AssertionError):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(self.argv("preflight")), 0)
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(initialize.call_count, 1)
        self.assertIn("preflight", output.getvalue())

    def test_run_and_resume_delegate_to_controlled_runner(self):
        for command in ("run", "resume"):
            with self.subTest(command=command):
                prepared = self.prepared()
                results = (SimpleNamespace(__dict__={"coordinate": "mock"}),)
                with patch("run_primary_v1_colab.controlled_runner.prepare_study", return_value=prepared), patch(
                    "run_primary_v1_colab.controlled_runner.initialize_study"
                ), patch("run_primary_v1_colab.controlled_runner.run_study", return_value=results) as run:
                    self.assertEqual(cli.main(self.argv(command)), 0)
                self.assertEqual(run.call_args.args, (prepared,))

    def test_status_validates_existing_binding_and_does_not_execute(self):
        prepared = self.prepared()
        prepared.study_directory.mkdir(parents=True)
        with patch("run_primary_v1_colab.controlled_runner.prepare_study", return_value=prepared), patch(
            "run_primary_v1_colab.controlled_runner.validate_study_directory"
        ) as validate, patch("run_primary_v1_colab.controlled_runner.run_study", side_effect=AssertionError), patch(
            "run_primary_v1_colab.controlled_runner.reconstruct_runtime_state",
            return_value={"derived_state": "PREFLIGHTED", "coordinates": [], "complete_coordinate_count": 0},
        ):
            self.assertEqual(cli.main(self.argv("status")), 0)
        self.assertEqual(validate.call_count, 1)

    def test_preflight_errors_from_binding_or_device_checks_fail_closed(self):
        prepared = self.prepared()
        for message in ("study binding mismatch", "gpu unavailable", "candidate count differs", "CTGAN feasibility failed"):
            with self.subTest(message=message), patch(
                "run_primary_v1_colab.controlled_runner.prepare_study", side_effect=ValueError(message)
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(self.argv()), 2)

    def test_unavailable_requested_ae_device_fails_preflight_environment_guard(self):
        fake_tensorflow = SimpleNamespace(config=SimpleNamespace(
            get_visible_devices=lambda: [],
            list_physical_devices=lambda kind: [],
        ))
        versions = {"numpy": "x", "pandas": "x", "scikit_learn": "x", "tensorflow": "x", "torch": "x", "sdv": "x", "ctgan": "x"}
        with patch("oncoassist_research.controlled_runner._package_versions", return_value=versions), patch(
            "oncoassist_research.controlled_runner.importlib.import_module", return_value=fake_tensorflow
        ):
            with self.assertRaisesRegex(RuntimeError, "no CPU"):
                controlled_runner._runtime_environment("cpu")
            with self.assertRaisesRegex(RuntimeError, "no TensorFlow GPU"):
                controlled_runner._runtime_environment("gpu")

    def test_recovery_requires_explicit_lock_identity(self):
        prepared = self.prepared()
        with patch("run_primary_v1_colab.controlled_runner.prepare_study", return_value=prepared), patch(
            "run_primary_v1_colab.controlled_runner.initialize_study"
        ), patch("run_primary_v1_colab.controlled_runner.recover_abandoned_study_lock") as recover:
            self.assertEqual(cli.main([*self.argv("recover-abandoned-lock"), "--expected-lock-id", "lock-123"]), 0)
        self.assertEqual(recover.call_args.args[2], "lock-123")


if __name__ == "__main__":
    unittest.main()
