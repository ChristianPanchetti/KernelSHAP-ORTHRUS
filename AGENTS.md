# Contributor constraints

KSHAP_ORTHRUS is functionally complete. Prefer small, focused changes; preserve
production module paths and supported APIs. Do not redesign stable components
or expand the project without an explicit task.

## Implemented contracts

- Real inference, Kernel SHAP orchestration, PostgreSQL/offline mapping, scanner,
  ground-truth export and offline visualizations are implemented.
- Real explanations use native `TemporalData` inside `OrthrusAlertCase`.
  `LogDataset` and the dummy adapter remain supported regression/debug tools.
- Real experiment grouping is source-node based, maximum eight components,
  with OTHER aggregation. Preserve canonical component/mask insertion order.
- Neutralization modifies only current `x_src`/`x_dst` occurrences, expanded
  independently across repeated nodes within the same role. Preserve history,
  topology, order, cardinality, messages, edge features and current edge targets.
- Historical neighbor-loader `e_id` values are global history IDs, never local
  batch row indices. Keep temporal preparation and snapshot/restore behavior.
- The real adapter validates per-edge losses and returns their scalar mean.
  Isolated mask evaluations are serial and restore state even after failures.
- `f(0)` is a neutralized reference state, not a benign baseline. Negative SHAP
  does not establish benign behavior or causal event attribution.
- Mapping supports readability, not inference, and happens outside mask scoring.
  Preserve missing/ambiguous identity and provenance handling. PostgreSQL remains
  an optional dependency with credentials supplied outside tracked source.
- Scanning advances state sequentially without SHAP or DB lookup. Ground-truth
  node membership is descriptive, not an official TP/FP/FN batch label.
- ORTHRUS source location is configured through `external_root`; do not hardcode
  a checkout path. Do not modify independent ORTHRUS code without explicit scope.

## Changes and verification

- Inspect existing behavior before editing. Preserve dummy compatibility.
- Do not change SHAP, perturbation, temporal-state or scoring semantics unless
  explicitly requested. Do not remove supported placeholder/compatibility APIs.
- Use synthetic tests; never require DARPA, checkpoints, PostgreSQL or external
  services in the portable suite. Run relevant checks after changes and report
  optional-dependency skips honestly.
- Real inference, preprocessing, training, scans and database experiments need
  task authorization; they are not routine cleanup verification.
- Keep research results and credentials out of commits. Do not delete ignored
  research outputs or independent checkouts merely because they are ignored.
- Keep technical documentation in the three maintained guides. Update claims
  against implementation and distinguish reported server evidence from new runs.
- Explainability of the reproduced instance does not establish equivalence with
  the published detector or its accuracy.

See [architecture](docs/framework_architecture.md),
[integration](docs/orthrus_integration.md) and
[experimental procedure](docs/experimental_pipeline.md).
