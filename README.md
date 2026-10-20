# KSHAP_ORTHRUS

## Overview

Kernel SHAP explanations for temporal provenance anomaly scores from a reproduced
ORTHRUS detector. The real adapter reduces per-edge losses to a mean scalar score;
component-level explanations preserve the target batch's temporal/topological
semantics. A separate dummy pipeline supports local regression checks.

## Architecture

```text
Temporal batch -> interpretable components -> binary masks
-> node-feature neutralization -> ORTHRUS inference -> edge losses
-> mean scalar score -> Kernel SHAP -> mapping/visualization
```

See [framework architecture](docs/framework_architecture.md).

## Main design choices

- Source-node grouping, maximum eight components, with remaining groups combined
  into one `OTHER` component when needed.
- Neutralize current `x_src`/`x_dst` occurrences across repeated nodes within each
  role; preserve topology, chronology, targets and shared historical `full_data`.
- Verify checkpoint history and prepare the official temporal prefix, then
  snapshot/restore pre-batch state for every serial SHAP evaluation.
- `f(0)` is a neutralized reference state, not a benign baseline. Negative SHAP
  lowers the score relative to that reference; it does not mean benign behavior.

## Requirements

Install the normal Python dependencies (NumPy, SHAP and matplotlib):

```bash
python -m pip install -r requirements.txt
```

Real inference additionally requires a compatible external ORTHRUS environment
with PyTorch/PyG, official YAML, preprocessed artifacts and a complete checkpoint
containing model weights and neighbor-loader state. Configure `external_root`
and artifact/checkpoint paths in the runtime JSON. No particular local checkout
location is required. The five final configurations under
`experiments/theia_e5/` use the recorded Linux `/home` layout; these are experiment
records, not portable defaults. The initial case is `initial_validation.json`.

PostgreSQL mapping additionally needs optional `psycopg2` and matching database
access through existing libpq environment/service settings. Keep credentials
outside the repository. Offline mapping uses verified `events`/`nodes` JSON rows
and still runs the real model. See [integration](docs/orthrus_integration.md).

## Reference ORTHRUS environment used for THEIA_E5

This is the verified experimental instance, not a generic installation requirement.
Real inference still depends on an externally configured source/artifact environment.
The ORTHRUS source base commit was `e7f25dfee1ddd182a955b88f8a90a8cbd4a8e543`,
with the two local patches described in [integration details](docs/orthrus_integration.md#local-source-patches).

| Runtime layout | Verified value |
| --- | --- |
| Docker container / KSHAP root | `orthrus-pids` / `/workspace/KSHAP` |
| ORTHRUS root / source | `/home` / `/home/src` |
| YAML / artifacts | `/home/config/orthrus.yml` / `/home/artifacts` |
| Pretrained weights | `/home/weights/THEIA_E5.pkl` |

Graph construction used `orthrus`, all files, 15.0 time windows and hashed labels.
Subject features were `type,path,cmd_line`; file features `type,path`; netflow
features `type,remote_ip,remote_port`. These are model-input settings, distinct
from the restrictions on publishing mapped network identities.

| Word2Vec | Verified settings |
| --- | --- |
| Method / embedding / seed | `feature_word2vec` / 128 / `use_seed=True` |
| Window / minimum count / skip-gram | 5 / 1 / `True` |
| Workers / epochs / negative samples | 6 / 50 / 5 |
| Node types / command / port | `False` / `True` / `False` |
| Decline rate | 30 |

| GNN and encoder | Verified settings |
| --- | --- |
| Training epochs / learning rate / weight decay | 6 / 0.00001 / 0.00001 |
| Hidden / output dimensions | 128 / 64 |
| Node type in node features / node features in GNN | `True` / `True` |
| Neighbor sampling / size | `[]` / 20 |
| Edge features / temporal dimension / batch size | `edge_type` / 100 / 1024 |
| Graph attention dropout / activation / heads | 0.5 / `relu` / 8 |

| Decoder and evaluation | Verified settings |
| --- | --- |
| Decoder / objective | Custom / `predict_edge_type` |
| Decoder dropout / layers / activation | 0.0 / 2 / `relu` |
| GNN testing threshold method | `str` |
| Ground truth / node threshold | `darpa` / `max_val_loss` |
| Destination loss / k-means / top K | `True` / `True` / 20 |

Attack reconstruction used `depimpact`, component mode, degree scoring, eight
workers and visualization disabled. These detector configuration settings are
recorded for provenance; KSHAP does not execute the complete evaluation or attack
reconstruction workflow when explaining a selected batch.

## Running a real explanation

From the repository root in the configured server environment:

```bash
python main.py --mode orthrus --adapter real \
  --orthrus-config experiments/theia_e5/initial_validation.json \
  --grouping-mode node \
  --perturbation-mode neutralize_edges \
  --max-components 8 \
  --num-samples 256 \
  --seed 0 \
  --mapping-backend postgresql \
  --output outputs/theia_e5/explanations/initial_validation/result.json \
  --log-file outputs/theia_e5/explanations/initial_validation/run.log
```

For offline metadata, use `--mapping-backend offline --mapping-rows path/to/rows.json`.
The device and selected graph/batch come from the runtime JSON. `--input` is a
legacy dummy option; real execution uses `--orthrus-config`.

## Full test-set scanner

```bash
python main.py --mode orthrus --adapter real \
  --orthrus-config experiments/theia_e5/initial_validation.json \
  --scan-test --scan-output outputs/theia_e5/scan/theia_e5_test_scan.csv \
  --scan-ground-truth outputs/theia_e5/scan/theia_e5_official_node_ground_truth.json \
  --log-file outputs/theia_e5/scan/theia_e5_test_scan.log
```

Scanning verifies checkpoint history and advances through validation and test
sequentially, once per batch, without SHAP, perturbations or database mapping.
It writes batch-loss statistics and a summary with representative high/low/median
cases for later manual explanation. It ignores single-case graph/batch selection.
The scan can be expensive and does not compute official detector metrics.

## Ground-truth export

Export official node IDs once in the matching server/database environment:

```bash
python scripts/export_scan_ground_truth.py \
  --orthrus-config experiments/theia_e5/initial_validation.json \
  --output outputs/theia_e5/scan/theia_e5_official_node_ground_truth.json
```

The exporter calls official `labelling.get_ground_truth(cfg)` without inference
or SHAP. The scanner uses this offline export without DB access. Node membership
is descriptive, not an official TP/FP/FN batch label. Verify that labels and
artifacts share the same node-ID namespace.

## Outputs

A real explanation creates:

- JSON: scores, signed component SHAP values, ranking, mapping and diagnostics.
- SHAP bar and waterfall: PNG/PDF, including reconstruction information.
- Provenance subgraph: PNG/PDF providing mapped context for attributed components.
- Summary table: CSV/PNG/PDF, retaining OTHER as an aggregate.

Use `--output-csv path/to/ranking.csv` for an additional ranking export.
Regenerate the nine visualization artifacts without inference, SHAP or DB jobs:

```bash
python main.py --plot-json outputs/theia_e5/explanations/initial_validation/result.json
```

Final artifacts are grouped under `outputs/theia_e5/scan/` and
`outputs/theia_e5/explanations/<case>/`. Explanation archives contain `result.json`,
`run.log`, `shap_bar.png/pdf`, `waterfall.png/pdf`, `provenance_subgraph.png/pdf`
and `summary_table.csv/png/pdf`. The final THEIA_E5 evidence is tracked in Git;
other generated outputs remain ignored. Exporter behavior is unchanged: new
plotting writes files prefixed by the JSON stem (`result_shap_bar.*`,
`result_shap_waterfall.*`, etc.); archived
figures have the shorter names above.

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The consolidated 15-file synthetic suite covers components/OTHER, repeated-role
neutralization, immutable history, state restoration, scalar scoring, official
temporal preparation, scanner sequencing, mapping/export, SHAP reconstruction
and reproducibility, visualizations and CLI behavior. Small synthetic SHAP
checks use the actual library with simulated inputs/models. Tests do not need
real ORTHRUS artifacts, PostgreSQL or external services; optional Torch/rendering
checks skip when dependencies are absent.

For the dummy/debug pipeline:

```bash
python main.py --mode dummy --adapter dummy --input logs_input.json \
  --output outputs/dummy.json
```

## Reproducibility limitation

Kernel SHAP explains the mean of per-edge reconstruction losses for the selected
batch. This scalar is not the complete official ORTHRUS node-level detection and
evaluation pipeline. The reproduced detector does not reproduce the published
E5-THEIA detector-level baseline. Explanations refer to the reproduced ORTHRUS
instance actually executed, not an exact reproduction of the detector in the
paper. This workflow does not establish detector accuracy. Record source patches, artifacts/checkpoint, environment, seed, device,
temporal context and mapping/label provenance; synthetic tests do not validate
server experiments. Detailed evidence and limitations are in the experiment guide.

## Repository structure

```text
KSHAP_ORTHRUS/
  main.py, pipeline.py, arg_parser.py, config.py, logger.py
  adapters/       # scoring and official/generic runtimes
  perturbation/   # interpretable builders and mask operations
  preprocessing/  # case loading, mapping and sidecars
  xai/            # SHAP, scanner, export and visualization
  scripts/        # official smoke launcher and ground-truth exporter
  experiments/theia_e5/  # five final configurations
  tests/          # 15 synthetic test files and shared fixtures
  docs/           # three maintained technical guides
  outputs/theia_e5/  # tracked final scan and five explanations
  report/         # standalone English LaTeX report and required figures
  logs_input.json
  requirements.txt, requirements-dev.txt
  LICENSE
```

## Documentation

- [Framework architecture](docs/framework_architecture.md)
- [ORTHRUS integration](docs/orthrus_integration.md)
- [Experimental procedure and reproducibility](docs/experimental_pipeline.md)
- [English project report](report/report.tex)

The English report is self-contained apart from its required `report/figures/`
resources. Compile it twice with a working LaTeX installation:

```bash
cd report
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Compilation uses saved figures and requires no detector or experimental jobs.

Licensed under MIT; see `LICENSE`.
