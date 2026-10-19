# Framework architecture

KSHAP_ORTHRUS explains a scalar anomaly score from an already reproduced
ORTHRUS detector. Real inference and Kernel SHAP orchestration are implemented.
The dummy pipeline remains a deterministic regression/debug facility; its
scores do not represent ORTHRUS detection behavior.

## Data flow and responsibilities

```text
official artifacts + checkpoint + temporal prefix
  -> OrthrusAlertCase(temporal_data, full_data)
  -> source-node components -> binary masks
  -> current node-feature neutralization -> state-isolated ORTHRUS forward
  -> per-edge losses -> mean scalar score -> Kernel SHAP
  -> one mapping enrichment -> JSON, ranking and visualizations
```

- `main.py`, `arg_parser.py`, `config.py` and `pipeline.py` provide CLI
  configuration and orchestration. Plotting and scanning dispatch separately.
- `adapters/orthrus_runtime.py` loads the configured external source, artifacts
  and checkpoint and prepares the official temporal context.
- `OrthrusAlertCase` separates the current batch from shared historical
  `full_data` and descriptive mapping metadata.
- `perturbation/` constructs components and applies masks.
- `RealOrthrusAnoAdapter` calls `model(batch, full_data, inference=True)` and
  validates per-edge loss cardinality before returning `sum(losses)/len(losses)`.
- `xai/kernel_shap_explainer.py` uses `shap.KernelExplainer`; exporters and
  visualizers preserve its signed component values.

The generic runtime and standalone artifact loader remain compatibility APIs.
The official pipeline constructs its case through coordinated runtime loading;
it does not treat the standalone loader as a complete temporal preparation.
Production modules keep their existing import paths and lazy external imports.

## Components and neutralization

The real experiment path uses `grouping_mode=node`, partitioning local batch
edges by **source node**, with a maximum of eight components. If necessary,
the largest groups are retained and the remainder forms one `OTHER` component.
OTHER receives one aggregate SHAP value. A maximum of one is accepted only
when the resulting space truly has one component.

`component_to_edges` maps component IDs to local batch edge positions. Its
insertion order defines mask positions and SHAP-value order; ranking changes
presentation only. The pipeline checks nonempty, disjoint assignments covering
the batch. Other builder groupings remain supported APIs, but are not the real
experiment CLI contract. The debug `time_chunk_seconds` value uses the numeric
unit of `t` without automatic conversion.

`neutralize_edges` (alias `mask_edge_features`) copies and zeros only current
`x_src` and `x_dst` feature occurrences. For every inactive edge, its source
node is neutralized at **all source occurrences**, and its destination node at
**all destination occurrences**. Expansion neither crosses roles nor recurses.
It can affect feature rows assigned to active components, so interventions
can overlap even though component edge assignments are disjoint.

Every mask starts from the original case. `src`, `dst`, `t`, `edge_index`,
current `edge_type`, `msg`, `edge_feats`, edge count/order and historical
`full_data` remain unchanged. Modified features preserve dtype and device.
`drop_edges` remains synthetic/debug or non-stateful functionality; local edge
removal does not preserve the official temporal loader's global indexing.

## Temporal isolation and interpretation

Preparation verifies the training checkpoint's neighbor memory and traverses
the official validation/test prefix before reaching the target batch. The
target has not been forwarded when the isolated adapter captures its snapshot.
Each mask restores that same state before inference and again in `finally`,
including failures. The snapshot covers neighbor-loader counters/IDs/neighbors,
available associations and GraphReindexer caches, including `None` caches.
Weights are not copied. This contract supports serial, non-contrastive inference.

Historical `LastNeighborLoader.e_id` values index global history; they are not
current-batch row indices, graph-edge indices or original event UUIDs.

`f(0)` is the neutralized reference state at fixed topology, history and
targets; it is **not a benign baseline**. Positive SHAP increases the explained
score relative to that reference; negative SHAP decreases it. Negative SHAP
does not mean benign behavior. These are component-level feature interventions,
not causal evidence about original events or independent event attributions.

These interventions act on extracted latent node features, including node-type
indicators when present; they do not mask raw commands/arguments or rerun
Word2Vec. All-zero latent features can be outside the training distribution.
The explanation remains conditional on retained history, which may still carry
information about the same nodes. Additive reconstruction alone does not prove
masker plausibility or attribution faithfulness.

The explainer shares a cached scorer across endpoints and coalitions and
exports expected value, SHAP sum, reconstruction residual, tolerance, seed,
library version and sampling diagnostics. It reports discrepancies rather
than changing attribution values to force reconstruction.

## Mapping and presentation boundary

Mapping enriches readability after SHAP, once for the component union; it is
not queried inside the scorer. Missing/ambiguous identities remain explicit.
The JSON retains canonical IDs, signed values, ranking, mapping provenance and
diagnostics. Legacy `num_records` denotes local edge counts on the real path,
as declared in result metadata. The provenance graph provides event/counterpart
context; those displayed entities do not gain separate SHAP attributions.

See [runtime and mapping contracts](orthrus_integration.md) and
[experimental procedure and limitations](experimental_pipeline.md).
