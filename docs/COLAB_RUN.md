# Controlled Primary V1 On Colab

This release runs the project source directly. It does not copy scientific code into notebook cells. Feature provenance remains `D. UNKNOWN`: the three supplied released/preselected modality matrices are required, but upstream feature-selection safety is not asserted.

## 1. Mount Persistent Storage

```python
from google.colab import drive
drive.mount('/content/drive')
```

Use a Google Drive directory for outputs. Colab runtime memory, including `/content`, is disposable after resets.

```bash
%cd /content/drive/MyDrive/OncoAssist_Research
```

## 2. Install Declared Dependencies

```bash
!python -m pip install -r requirements-colab.txt
```

The requirements file contains bounded versions validated against the project's imports. It does not silently alter dependencies from production code. The preflight command is authoritative for the final TensorFlow, PyTorch, SDV, and CTGAN compatibility check.

## 3. Declare Inputs And Run Identity

Set each path explicitly; the runner will not scan a directory or choose arbitrary CSV files.

```bash
export MGE_CSV=/content/drive/MyDrive/blca/mGE.csv
export MDM_CSV=/content/drive/MyDrive/blca/mDM.csv
export MCNA_CSV=/content/drive/MyDrive/blca/mCNA.csv
export OUTPUT_DIR=/content/drive/MyDrive/oncoassist_primary_v1_runs
export RUN_ID=primary-v1-colab-001
export ROOT_SEED=20260823
export AE_DEVICE=gpu
```

Choose `AE_DEVICE=cpu` or `AE_DEVICE=gpu` explicitly. A GPU request fails if TensorFlow reports no physical GPU. CTGAN is always run in its isolated CPU-only worker; GPU availability does not change that policy.

## 4. Run Preflight

```bash
!python run_primary_v1_colab.py preflight \
  --mge "$MGE_CSV" --mdm "$MDM_CSV" --mcna "$MCNA_CSV" \
  --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --root-seed "$ROOT_SEED" \
  --ae-device "$AE_DEVICE"
```

Preflight performs no fitting, scoring, sampling, or study execution. It validates source paths, canonical alignment, protocol/provenance/content identities, 25 outer coordinates, 3 inner folds per coordinate, candidate grid, CTGAN PAC feasibility, worker constructor compatibility without `.fit()` or sampling, output filesystem semantics, device policy, and the immutable reference SHA.

For a new study it creates the immutable study binding, preflight record, and fold manifests under `$OUTPUT_DIR/<study-identity>/`.

## 5. Inspect Status

```bash
!python run_primary_v1_colab.py status \
  --mge "$MGE_CSV" --mdm "$MDM_CSV" --mcna "$MCNA_CSV" \
  --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --root-seed "$ROOT_SEED" \
  --ae-device "$AE_DEVICE"
```

The JSON output includes study/provenance/protocol/reference identities, device policy, completed coordinates, evaluation-only resumes, remaining coordinates, and failure-record count. It never prints biological matrices or synthetic rows.

## 6. Start Or Resume The Study

```bash
!python run_primary_v1_colab.py run \
  --mge "$MGE_CSV" --mdm "$MDM_CSV" --mcna "$MCNA_CSV" \
  --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --root-seed "$ROOT_SEED" \
  --ae-device "$AE_DEVICE"
```

After a disconnect or runtime restart, rerun the exact same command with `resume` instead of `run`:

```bash
!python run_primary_v1_colab.py resume \
  --mge "$MGE_CSV" --mdm "$MDM_CSV" --mcna "$MCNA_CSV" \
  --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --root-seed "$ROOT_SEED" \
  --ae-device "$AE_DEVICE"
```

Resume reruns preflight and accepts only the exact immutable binding. Completed coordinates are skipped. A valid scoring publication without evaluation is evaluated only, without rebuilding models or rescoring. Any incomplete scoring coordinate is recomputed from the public inner-search stage; no in-memory or partial scientific state is reused.

## 7. Explicit Abandoned Lock Recovery

Colab host changes cannot be auto-classified as safe stale-lock recovery. Confirm no other run is active, obtain the exact lock ID from `.run_lock.json`, then run:

```bash
!python run_primary_v1_colab.py recover-abandoned-lock \
  --mge "$MGE_CSV" --mdm "$MDM_CSV" --mcna "$MCNA_CSV" \
  --output-dir "$OUTPUT_DIR" --run-id "$RUN_ID" --root-seed "$ROOT_SEED" \
  --ae-device "$AE_DEVICE" --expected-lock-id "LOCK_ID"
```

This action records immutable abandonment lifecycle evidence. It never silently breaks a remote lock.

## 8. Completion And Artifacts

Completion requires all 25 coordinates to have valid scoring and evaluation publications. Per-coordinate outputs are under:

```text
$OUTPUT_DIR/<study-identity>/outer_folds/repeat-XX/fold-XX/scoring/
$OUTPUT_DIR/<study-identity>/outer_folds/repeat-XX/fold-XX/evaluation/
```

The `evaluation/evaluation.json` files contain machine-readable future aggregation inputs. Statistical aggregation and paper-result generation are separate work and are not performed by this release command.
