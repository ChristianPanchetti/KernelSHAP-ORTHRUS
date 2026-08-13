# AGENTS.md

## Role

You are working on the KSHAP_ORTHRUS thesis project.

The goal is to build an explainability framework for ORTHRUS-ano using Kernel SHAP.

Always prefer small, safe, testable changes. Do not rewrite working modules unless explicitly requested.

## Main target pipeline

The final target pipeline is:

DARPA / ORTHRUS artifacts
→ TemporalData
→ OrthrusAlertCase
→ OrthrusInterpretableBuilder
→ Kernel SHAP masks
→ OrthrusPerturbationManager.neutralize_edges
→ RealOrthrusAnoAdapter
→ ORTHRUS edge losses
→ scalar anomaly score
→ SHAP values
→ readable explanation.

The project is now moving toward real ORTHRUS inference. Do not spend time implementing unnecessary alternative pipelines unless explicitly requested.

## Architectural rules

- `LogDataset` and `DummyOrthrusAnoAdapter` are legacy/debug mode.
- Dummy mode must keep working.
- The final ORTHRUS pipeline must use ORTHRUS `TemporalData`, not `LogDataset`.
- `OrthrusAlertCase` is the central object for ORTHRUS explanations.
- SHAP works on interpretable components mapped to edge indices through `component_to_edges`.
- `neutralize_edges` (also exposed as `mask_edge_features`) is the preferred real-ORTHRUS perturbation strategy because it preserves chronological edge order and global `e_id` alignment.
- `drop_edges` is retained for debug/synthetic use, non-stateful models, or a future integration that manages global `e_id` explicitly.
- ORTHRUS inference is expected to return per-edge losses.
- The adapter must reduce edge losses to one scalar score.
- Initial scalar reduction should be `mean(edge_losses)` unless explicitly changed.
- Mapping providers are for explanation readability, not for model inference.
- `ArtifactOnlyMappingProvider` is a fallback/debug mapping.
- `SidecarMappingProvider` is useful only if sidecar files exist.
- `DbAssistedMappingProvider` is future work unless explicitly requested.

## Implementation priorities

Prefer this order:

1. Audit existing code before modifying it.
2. Keep existing tests passing.
3. Obtain or generate real ORTHRUS artifacts.
4. Run the official unperturbed smoke test on real `TemporalData` and a complete checkpoint.
5. Validate `full_data`, global `e_id`, neighbor-loader state, and device behavior.
6. Connect the validated runtime to the ORTHRUS Kernel SHAP path.
7. Improve mapping/export only after real inference works.

Do not prioritize:
- extra mapping strategies;
- visualization;
- advanced reporting;
- CSV/JSONL sidecar variants;
- real Postgres integration;
- GraphAssistedMappingProvider;
- complex export formats.

## Coding rules

- Make minimal focused changes.
- Do not modify Kernel SHAP core logic unless explicitly requested.
- Do not modify external ORTHRUS code unless strictly necessary.
- Do not require DARPA, Postgres, or real ORTHRUS artifacts in unit tests.
- Use synthetic tests where possible.
- Do not introduce hard dependencies on Postgres.
- Preserve backward compatibility unless explicitly told otherwise.
- If a task is ambiguous, inspect the code and explain the safest option before implementing.
- If an interrupted previous implementation exists, audit it first and complete only missing parts.

## Testing rules

Before final response, run relevant tests.

Prefer:
1. targeted tests for changed files;
2. then full `pytest -q` if feasible.

Add tests for any new behavior.

Tests must not require:
- DARPA real dataset;
- Postgres;
- real ORTHRUS checkpoint;
- external services.

## ORTHRUS-specific cautions

Be careful with:
- `TemporalData` required fields;
- `src`, `dst`, `t`, `msg`;
- optional `edge_type`, `edge_index`, `x_src`, `x_dst`, `edge_feats`;
- `full_data`;
- `e_id`;
- neighbor loaders / temporal loaders;
- shape alignment after edge dropping.

Never assume that perturbing the local batch is enough until compatibility with `full_data` and `e_id` has been checked.

`RealOrthrusAnoAdapter`, the generic runtime, the official cfg-driven runtime, and the smoke-test CLI are implemented and synthetically tested. The next practical block is obtaining or generating real ORTHRUS artifacts and running the official unperturbed smoke test. Kernel SHAP end-to-end in ORTHRUS mode is still future work.

## Sidecar and mapping

The sidecar format, if used, maps TemporalData edge indices to original event metadata:

```json
{
  "mapping_mode": "sidecar-assisted",
  "join_strategy": "edge_index",
  "edges": {
    "0": {
      "event_uuid": "...",
      "src_index_id": 123,
      "dst_index_id": 456,
      "timestamp_rec": 123456789,
      "operation": "EVENT_EXECUTE",
      "raw_metadata": {}
    }
  }
}
```
