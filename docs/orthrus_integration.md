# ORTHRUS integration

## External runtime contract

ORTHRUS source is configured through `external_root`; its location is independent
of this repository. The five final JSON configurations under
`experiments/theia_e5/` describe the verified server/container layout under
`/home`; review those paths before use. Relative runtime JSON
paths resolve from the launch directory, not the JSON's parent directory.

Real execution needs the compatible ORTHRUS source and its Python environment,
PyTorch/PyG, preprocessed `.TemporalData.simple` artifacts, official YAML and a
complete `model_epoch_N` checkpoint with `state_dict.pkl` and
`neighbor_loader.pkl`. These are not bundled in the tracked framework.
`config_path` must identify `<external_root>/config/orthrus.yml`, because the
official configuration loader uses that file. Overrides follow official dotted
configuration keys or CLI arguments.

The runtime loads official config/data/factory modules from the configured
source tree. It temporarily changes working directory to `external_root` for
official `./artifacts` resolution and restores the caller's directory in
`finally`. `load_all_datasets` supplies train, validation, test, shared
`full_data` and maximum node count; `build_model`, `load_model` and the official
batch-loader factory prepare inference. Optional `from_weights_path` replaces
weights after checkpoint loading but does not replace neighbor memory.

## Verified THEIA_E5 experimental environment

The following is the verified server configuration actually used for THEIA_E5
experiments, not a universal KSHAP requirement. Source/artifact locations remain
externally configured. The source base revision identifies the starting code;
the executed instance also includes the two local patches below.

| Runtime item | Verified value |
| --- | --- |
| Docker container | `orthrus-pids` |
| KSHAP working directory | `/workspace/KSHAP` |
| `external_root` | `/home` |
| ORTHRUS source directory | `/home/src` |
| `config_path` | `/home/config/orthrus.yml` |
| Artifacts | `/home/artifacts` |
| `from_weights_path` | `/home/weights/THEIA_E5.pkl` |
| ORTHRUS source base commit | `e7f25dfee1ddd182a955b88f8a90a8cbd4a8e543` |

`external_root` is the source tree's parent of `src` and `config`, rather than
`/home/src` itself. The runtime JSON separately selects the complete local
`model_epoch_N` checkpoint. Loading first restores its `state_dict.pkl` and
`neighbor_loader.pkl`; the pretrained weights above then replace the parameter
state without replacing that local neighbor state. Thus the pretrained weight
file alone is insufficient to reproduce the temporal execution. Record the
checkpoint, artifacts and selected batch together; verify end-of-train neighbor
history before advancing the official evaluation prefix.

### Graph construction

| Setting | Verified value |
| --- | --- |
| `used_method` | `orthrus` |
| `use_all_files` | `True` |
| `time_window_size` | `15.0` |
| `use_hashed_label` | `True` |
| Subject features | `type`, `path`, `cmd_line` |
| File features | `type`, `path` |
| Netflow features | `type`, `remote_ip`, `remote_port` |

### Word2Vec

| Setting | Verified value |
| --- | --- |
| `method` / `emb_dim` / `use_seed` | `feature_word2vec` / 128 / `True` |
| `window_size` / `min_count` / `skip_gram` | 5 / 1 / `True` |
| `num_workers` / `epochs` / `negative` | 6 / 50 / 5 |
| `use_node_types` / `use_cmd` / `use_port` | `False` / `True` / `False` |
| `decline_rate` | 30 |

### GNN training and encoder

| Setting | Verified value |
| --- | --- |
| `num_epochs` / `lr` / `weight_decay` | 6 / 0.00001 / 0.00001 |
| `node_hid_dim` / `node_out_dim` | 128 / 64 |
| `use_node_type_in_node_feats` | `True` |
| `neighbor_sampling` / `neighbor_size` | `[]` / 20 |
| `edge_features` / `use_node_feats_in_gnn` | `edge_type` / `True` |
| `temporal_dim` / `batch_size` | 100 / 1024 |
| Graph attention dropout / activation / `num_heads` | 0.5 / `relu` / 8 |

### Decoder, testing and evaluation

| Setting | Verified value |
| --- | --- |
| Decoder / `objective` | Custom / `predict_edge_type` |
| Decoder dropout / `num_layers` / activation | 0.0 / 2 / `relu` |
| `gnn_testing.threshold_method` | `str` |
| Evaluation `ground_truth_version` | `darpa` |
| `node_evaluation.threshold_method` | `max_val_loss` |
| `use_dst_node_loss` | `True` |
| `use_kmeans` / `kmeans_top_K` | `True` / 20 |

### Attack reconstruction

| Setting | Verified value |
| --- | --- |
| `method` / depimpact method | `depimpact` / `component` |
| `score_method` / workers / `visualize` | `degree` / 8 / `False` |

These settings describe the reproduced detector environment. Recording evaluation
and reconstruction configuration does not mean a KSHAP explanation or scanner
executes those complete official workflows.

## Local source patches

The verified source differs from its base commit in two relevant local patches.
They are documented here for reproducibility; this documentation update does not
apply or modify either patch in any ORTHRUS checkout.

### Patch 1: `src/data_utils.py`

This is primarily a memory/storage compatibility patch for the active
`edge_features=edge_type` configuration. It:

1. Inspects the configured edge features.
2. Avoids concatenating/storing `full_data.msg` when messages are not required
   on that inference path.
3. Still concatenates `full_data.t` and `full_data.edge_type` in original order.
4. Clones `edge_type` from the serialized message fields.
5. Constructs the current `msg` from `x_src`/`x_dst` for `predict_edge_type`.
6. Exposes `x_src`/`x_dst` as slices of that constructed `msg`.

The patch does not intentionally change the numerical feature values,
timestamps, edge types or ordering consumed in the tested `edge_type` path.
It is not claimed to be bit-identical internally to upstream storage: cloning,
construction and slicing change storage/aliasing details. Omitting historical
`full_data.msg` is valid here because that tensor is not consumed on this path;
it must not be generalized to arbitrary feature configurations. Framework
history alignment checks retain `t` and `edge_type` without reconstructing the
omitted tensor. KSHAP perturbations still preserve history and current targets.

### Patch 2: `src/orthrus.py`

Before the `run_from_training` conditional, the patch initializes:

```python
t1 = t2 = t3 = t0
```

This prevents uninitialized timing variables when preprocessing is skipped.
It affects timing/reporting only and does not alter model training or inference.

The smoke script is retained because the real explanation pipeline, scanner and
ground-truth exporter import its shared `load_config` and `build_config` helpers.
Its CLI remains an optional diagnostic interface, not a prerequisite extra run
before each final explanation.

## Temporal preparation and scoring

Real explanations require an end-of-training checkpoint, val/test selection,
fresh GraphReindexer caches and official evaluation order. Preparation checks
the checkpoint's counter and reconstructs training topology in a separate CPU
neighbor loader to compare historical IDs and neighbors. This is not training
or train-model inference, but still has memory/time cost.

The model then advances once per official prefix batch, validation before test,
checking current batch `t` and `edge_type` against global history at each offset.
Preparation returns before forwarding the chosen target. One isolated adapter
then scores all target masks from that pre-batch state. Full history remains
unchanged and on its original device; the official path indexes it using CPU
history IDs. The selected batch/model/loader use the configured device.

The default smoke launcher performs a direct unperturbed batch check. Its
success alone does not establish faithful temporal preparation. The supported
`--phase9` option performs preparation and A1/B/A2/Z isolated-mask validation;
normal explanation preparation returns without those extra four evaluations.

```bash
python scripts/run_orthrus_official_smoke.py path/to/runtime.json
python scripts/run_orthrus_official_smoke.py path/to/runtime.json --phase9
```

The scalar explained by Kernel SHAP is the mean of per-edge reconstruction
losses for the selected batch. It is not the complete official ORTHRUS node-level
detection/evaluation pipeline, whose aggregation, thresholds and labels have
separate semantics. `f(0)` is a neutralized reference state, not a benign baseline;
negative SHAP lowers this batch score relative to the reference and does not
mean benign behavior.

## Mapping contracts

Inference does not require human-readable identities. The explanation pipeline
does require its selected metadata backend to succeed before exporting a new
complete result. PostgreSQL enrichment uses optional `psycopg2` and existing
libpq environment/service settings; credentials do not belong in tracked JSON.
Offline mapping accepts a JSON object containing `events` and `nodes` arrays.
Those arrays must include all candidates for the requested keys to establish
uniqueness; offline mapping does not simulate inference.

Node lookup uses `index_id` and reports resolved, missing, ambiguous or
unresolvable identities. Event joins use oriented `(src_index_id, dst_index_id,
timestamp_rec, operation)` tuples. THEIA_E5 timestamps remain integer
nanoseconds; database orientation is not reversed again. SQL row IDs, event
UUIDs, graph offsets, local edge positions and historical `e_id` are distinct.
Official one-hot relation vectors require the supplied one-based `rel2id`
vocabulary; no vocabulary is guessed from argmax alone.

Database fetching uses selective parameterized queries, deduplicated keys,
read-only transactions and a statement timeout. Actual schema/index suitability
and database/artifact namespace correspondence must be checked on the server.
Network fields remain withheld unless explicitly verified: the inspected
THEIA_E5 upstream parser had a branch assigning UUID characters to endpoint
fields, so an identity lookup alone cannot certify network endpoint values.

Artifact-only mapping remains a fallback API. Sidecar generation/loading remain
optional APIs. Verified sidecars bind ordered graph keys to an artifact hash and
explicit graph-to-batch offset; that offset is not the global temporal offset.
Duplicate keys stay ambiguous. Legacy sidecars and joins lacking operation are
unverified; their UUIDs are not reused as verified database identities. Hashes
check artifact consistency, not source authenticity. Graph-assisted mapping
remains an explicit unsupported API.

See [architecture](framework_architecture.md) for perturbation/isolation and
[experiments](experimental_pipeline.md) for commands, labels and evidence.

## Reproducibility limitation

The reproduced ORTHRUS detector does not reproduce the published E5-THEIA
detector-level baseline. Explanations are faithful to the reproduced instance
actually executed, including the local patches, artifact/checkpoint combination
and temporal context; they are not explanations of an exact reproduction of the
detector in the paper. Neither the selected-batch mean nor descriptive scanner
node membership establishes published detector accuracy or official TP/FP/FN
batch labels. Preserve the actual runtime configuration and provenance when
interpreting or comparing experiments.
