# Experimental procedure and reproducibility

## Prepare and select cases

Use the reproduced detector's existing artifacts and complete checkpoint.
Record the external source revision and local patches, official configuration,
artifact/checkpoint identity, optional weight override, environment versions,
device and database namespace. Do not regenerate artifacts merely to explain
an existing score. The tracked THEIA_E5 JSONs select distinct test cases:

| Configuration in `experiments/theia_e5/` | Graph | Batch |
| --- | ---: | ---: |
| `initial_validation.json` | 0 | 0 |
| `high_malicious.json` | 40 | 0 |
| `high_no_malicious.json` | 35 | 82 |
| `low_malicious.json` | 98 | 39 |
| `median_reference.json` | 42 | 35 |

The filenames describe selected cases using node membership, not certified
batch detector labels. Paths target the recorded `/home` server layout and must
be checked against the actual environment. The JSON device controls inference.

## Export official node membership

Run this utility only in the matching official server/database environment:

```bash
python scripts/export_scan_ground_truth.py \
  --orthrus-config experiments/theia_e5/initial_validation.json \
  --output outputs/theia_e5/scan/theia_e5_official_node_ground_truth.json
```

The exporter calls official `labelling.get_ground_truth(cfg)` using the same
configuration and database selection. It exports exact, sorted, deduplicated
integer node IDs with dataset, label source/version/files and count. It performs
official database UUID lookup, but no model construction, inference or SHAP.
Do not substitute UUIDs or invent an empty label set when labels are unavailable.

## Scan the full test set

```bash
python main.py --mode orthrus --adapter real \
  --orthrus-config experiments/theia_e5/initial_validation.json \
  --scan-test --scan-output outputs/theia_e5/scan/theia_e5_test_scan.csv \
  --scan-ground-truth outputs/theia_e5/scan/theia_e5_official_node_ground_truth.json \
  --log-file outputs/theia_e5/scan/theia_e5_test_scan.log
```

The scanner verifies checkpoint history, forwards validation then all test
batches in official order, and advances temporal state once per batch without
snapshot restoration. It ignores single-case split/graph/batch selections.
It runs no SHAP, perturbation, metadata enrichment, visualization or database
lookup. Scanning still loads the normal runtime artifacts and can be expensive.

CSV rows include coordinates, edge count, mean/min/median/p95/max loss and
descriptive unique malicious-node membership counts. The mean uses the same
Python-float aggregation as the adapter. Node membership is **not an official
TP/FP/FN batch label**, temporal attack-event label or malicious-edge label.
Official detector metrics require the separate official evaluation workflow,
including its aggregation, thresholds and time-window loss artifacts.

The scanner checks export version and configured label files and records SHA-256
and provenance. These checks cannot prove the node-ID namespace matches the
temporal artifacts; establish that correspondence on the server.

The summary supplies deterministic high/low/median representative candidates
for later manual SHAP case selection. Ties follow scan order, missing categories
are explicit and categories may repeat a case. No SHAP run starts automatically.
CSV rows flush incrementally; a new scan removes an old summary, and a successful
summary appears only after completion. Interrupted scans require restarting
from the verified checkpoint: CSV is not a temporal-state checkpoint.

## Explain and inspect

```bash
python main.py --mode orthrus --adapter real \
  --orthrus-config experiments/theia_e5/initial_validation.json \
  --grouping-mode node --perturbation-mode neutralize_edges \
  --max-components 8 --num-samples 256 --seed 0 \
  --mapping-backend postgresql \
  --output outputs/theia_e5/explanations/initial_validation/result.json \
  --log-file outputs/theia_e5/explanations/initial_validation/run.log
```

For verified offline metadata, replace the mapping option with
`--mapping-backend offline --mapping-rows path/to/rows.json`.
Retain the original JSON, runtime configuration and provenance alongside the
case coordinates and environment record. Check finite scores/attributions,
canonical component order, OTHER aggregation, mapping quality and reconstruction
residual. The seed and budget do not guarantee identical results across library
versions or devices. Preparation and snapshot copies also contribute to cost.

Each real explanation writes JSON plus nine visual artifacts: bar and waterfall
PNG/PDF, provenance subgraph PNG/PDF, and summary table CSV/PNG/PDF. Optional
`--output-csv` additionally exports the attribution ranking. JSON retains full
labels/precision; display labels may be shortened. Regenerate visuals offline:

```bash
python main.py --plot-json outputs/theia_e5/explanations/initial_validation/result.json
```

This reads the result without model/SHAP/database execution. The provenance graph
shows up to five components and two counterparts per component; the summary
retains all components. Context events/counterparts have no separate attribution.
Read signed values relative to the neutralized `f(0)` reference, never as a
benign/malicious classification. See [architecture](framework_architecture.md).

## Final artifact archive

`outputs/theia_e5/scan/` retains the full scan CSV, summary JSON, official node
export and dedicated scan log. `outputs/theia_e5/explanations/<case>/` retains
`result.json`, `run.log`, `shap_bar.png/pdf`, `waterfall.png/pdf`,
`provenance_subgraph.png/pdf` and `summary_table.csv/png/pdf` for each of the five
cases. Result and scan payloads are preserved byte-for-byte; embedded historical
paths still identify the original export/run locations rather than the relocated
files. The initial-validation log is the original final-run session extracted
from the former aggregate log, not a newly executed run.

The initial-validation summary CSV was checked against all eight rows derived
from its final JSON; the provenance/summary PNGs and PDF text were checked for
matching components, signed values, identities and counts before relocation.
No inference, SHAP or plot regeneration was used for this check. Figures were
renamed for the archive only. The unchanged plot exporter creates JSON-stem
prefixed filenames when invoked again; it does not overwrite the short archive
names automatically.

Saved final scores below are read from the archived JSONs, not new measurements:

| Case | Original batch score | Neutralized reference |
| --- | ---: | ---: |
| initial_validation | 0.717194598622882 | 0.8296128102228977 |
| high_malicious | 1.7857131471537286 | 2.0763393884990364 |
| high_no_malicious | 1.7820469052912813 | 1.8329053156267037 |
| low_malicious | 0.04784710962121608 | 0.06480291016578121 |
| median_reference | 0.4125550483149709 | 1.1753264604049036 |

The initial result records SHAP 0.48.0, 254 internal coalitions, 256 unique masks,
eight components and seed 0. Runtime dependency lower bounds do not pin those
versions. `num_samples` is a coalition budget, not total detector cost: temporal
prefix preparation and snapshot restoration contribute separately. Record actual
SHAP version/default regularization and environment for comparisons; the seed
alone does not establish Torch/CUDA determinism.

## Evidence and limitations

The prior project record reports server validation on September 21, 2026 for
THEIA_E5 test/0/0, CUDA, 1,024 edges, global offset 16,351,122, 12,385,364 training
edges and 3,921 prefix batches. Reported isolated scores were:

| Evaluation | Score |
| --- | ---: |
| A1, all active | 0.7171946001362812 |
| B, `node:1301260` inactive | 0.7381046357841115 |
| A2, all active again | 0.7171945968802902 |
| Z, all selected components inactive | 0.8296128124493407 |

Each returned 1,024 losses; A1/A2 matched with `rtol=1e-6`, `atol=1e-8`, with
reported state/data invariants satisfied. This is retained reported evidence,
not a new execution or a table of SHAP values. The older direct-smoke score
0.7162450345895195 lacks the same documented pre-batch context and should not
be treated as an interchangeable reference. Existing output files alone do not
certify every server dependency or database mapping contract.

The explainability pipeline is faithful to the reproduced ORTHRUS instance
actually executed. The reproduced detector does not reproduce the published
E5-THEIA detector-level baseline and is not an exact reproduction of the detector
in the paper. Do not infer published accuracy from
this explanation workflow or replace official evaluation with batch means.
Upstream also documents Word2Vec reproducibility sensitivity to
`PYTHONHASHSEED`; preserve the actual environment and preprocessing provenance.

Synthetic tests establish local framework contracts, not detector accuracy,
CUDA reproducibility, official label namespace identity or live database
compatibility. No real experiment is required for repository cleanup.
