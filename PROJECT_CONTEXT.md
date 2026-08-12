# PROJECT_CONTEXT.md

## Project overview

This project implements an explainability framework for anomaly detection on provenance/process-log data.

The thesis objective is to explain anomaly scores produced by a graph-based anomaly detector, with focus on ORTHRUS-ano and DARPA E3/E5 provenance data.

The explanation method is based on Kernel SHAP. The framework builds interpretable components, perturbs them, queries the anomaly detector, and assigns importance values to components.

The final goal is not only to output SHAP values, but to produce analyst-facing explanations such as:

- which edge types mattered;
- which nodes/processes mattered;
- which temporal window mattered;
- which subgraph or set of events contributed to the anomaly;
- which original event metadata can be associated with important edges.

## Original Version 1

Version 1 used a simplified pipeline:

logs_input.json
→ LogDataset
→ InterpretableGraphBuilder
→ PerturbationManager
→ DummyOrthrusAnoAdapter
→ Kernel SHAP
→ SHAP values.

This version validated the general idea:

1. build interpretable components;
2. generate binary SHAP masks;
3. perturb the input;
4. query an adapter;
5. explain the score.

The dummy adapter returns artificial anomaly scores and is only for debugging/sanity checks.

## Why the pipeline changed

After analyzing ORTHRUS, the project changed direction.

ORTHRUS-ano does not consume `LogDataset`.

ORTHRUS consumes PyTorch Geometric-style `TemporalData` artifacts built from DARPA provenance data.

Therefore, the final pipeline should not force `LogDataset` into ORTHRUS. Instead, it should work directly with ORTHRUS artifacts.

The final ORTHRUS branch should use:

DARPA E3/E5
→ ORTHRUS preprocessing
→ provenance graphs
→ TemporalData
→ ORTHRUS-ano inference
→ edge losses
→ scalar anomaly score
→ Kernel SHAP explanation.

## ORTHRUS data model

DARPA events describe system-level provenance activities, such as:

- process creates another process;
- process reads a file;
- process writes a file;
- process opens a socket;
- process communicates over the network.

ORTHRUS organizes these events into provenance graphs.

In a provenance graph:

- nodes represent system entities, such as processes, files, sockets, netflows;
- edges represent typed interactions between entities;
- timestamps represent when interactions happened;
- attributes describe nodes and edges.

ORTHRUS then converts these graphs into `TemporalData`.

Important TemporalData fields include:

- `src`: source node ids;
- `dst`: destination node ids;
- `t`: timestamps;
- `msg`: edge/message features.

Additional derived fields may include:

- `edge_type`;
- `edge_index`;
- `x_src`;
- `x_dst`;
- `edge_feats`;
- `e_id`.

## OrthrusAlertCase

`OrthrusAlertCase` is the central object for ORTHRUS explanations.

It represents one local case to explain, such as:

- one anomalous window;
- one batch;
- one localized alert;
- one subgraph.

It may contain:

- `temporal_data`;
- `full_data`;
- `metadata`;
- `component_to_edges`;
- mapping dictionaries;
- optional legacy `log_dataset`.

It has been implemented structurally and tested mainly with synthetic/FakeTemporalData examples.

It still needs to be tested with real ORTHRUS TemporalData artifacts.

## Interpretable components

Kernel SHAP needs binary interpretable components.

For ORTHRUS mode, `OrthrusInterpretableBuilder` creates components over TemporalData edges.

Possible grouping modes:

- single edge;
- node;
- edge type;
- time chunk.

Example:

edge_type:EVENT_EXECUTE → edges [0, 2, 5]
time_chunk:3 → edges [20, 21, 22]
node:123 → edges [4, 8, 9]

SHAP masks turn these components on/off.

## ORTHRUS perturbation

The preferred perturbation for real ORTHRUS inference is `neutralize_edges` (also available as `mask_edge_features`). It preserves the number, order and topology of edges while zeroing edge-aligned feature fields for inactive components.

This decision follows the real inference contract: `LastNeighborLoader` assigns global `e_id` values through chronological insertion, and encoders may retrieve edge features from `full_data`. Dropping local edges can shift those identifiers or break alignment even when the sliced local batch looks structurally valid.

`drop_edges` remains implemented for synthetic/debug use, non-stateful models, and a possible future path with explicit global-`e_id` support.

For `drop_edges`, given a SHAP mask:

1. active components are selected;
2. active components are converted to edge indices through `component_to_edges`;
3. inactive edges are removed;
4. edge-aligned fields in TemporalData are filtered.

Fields to filter may include:

- `src`;
- `dst`;
- `t`;
- `msg`;
- `edge_type`;
- `x_src`;
- `x_dst`;
- `edge_feats`.

`edge_index` is rebuilt if possible.

The perturbation preserves traceability to original edge indices for explanation purposes.

Both dropping and neutralization are implemented and tested on synthetic data. Neutralization also has a Torch tensor test for dtype/device preservation.

It still needs compatibility checks with real ORTHRUS inference, especially for:

- `full_data`;
- `e_id`;
- neighbor loaders;
- temporal loaders.

## Mapping problem

SHAP explains components and edge indices.

However, edge indices are not always human-readable.

The desired final explanation should map important edges back to original events, such as:

- event_uuid;
- operation;
- process;
- file path;
- command line;
- timestamp;
- source/destination entities.

ORTHRUS keeps `event_uuid` in the database and/or intermediate graph construction, but it is not preserved in the final TemporalData artifact.

Therefore, mapping strategies were introduced.

## Mapping providers

Implemented or planned mapping providers:

### ArtifactOnlyMappingProvider

Uses only information available in TemporalData.

Example output:

edge 17:
src=123
dst=456
t=987654321
edge_type=EVENT_EXECUTE

This is a fallback/debug mapping.

### SidecarMappingProvider

Reads a sidecar JSON file mapping TemporalData edge indices to original metadata.

Example:

edge 17 → event_uuid abc-123, operation EVENT_EXECUTE, path /bin/bash

This is implemented and testable if a sidecar exists.

Automatic sidecar generation and validation are implemented and synthetically tested. They are useful for explanation readability but are not a blocker for model inference.

### DbAssistedMappingProvider

Future strategy.

It should reconstruct:

TemporalData edge → DB event row → event_uuid

using join keys such as:

(src, dst, timestamp, operation)

Current state: offline/testable skeleton only. No real Postgres dependency should be added unless explicitly requested.

## Current implementation state

Implemented:

- legacy LogDataset pipeline;
- DummyOrthrusAnoAdapter;
- Kernel SHAP wrapper for dummy pipeline;
- OrthrusAlertCase;
- OrthrusInterpretableBuilder;
- OrthrusPerturbationManager.drop_edges;
- OrthrusPerturbationManager.neutralize_edges / mask_edge_features;
- ArtifactOnlyMappingProvider;
- SidecarMappingProvider;
- DbAssistedMappingProvider offline skeleton;
- orthrus_join_keys utilities;
- synthetic tests for mapping and perturbation;
- sidecar generation and validation.

Partially implemented or scaffolded:

- OrthrusAlertCaseLoader (artifact loading implemented, real ORTHRUS artifact compatibility not yet validated);
- RealOrthrusAnoAdapter (core inference/reduction implemented via injected model; checkpoint/model construction not implemented);
- ORTHRUS end-to-end mode;
- db-assisted real mapping.

Not yet implemented:

- real DARPA E3/E5 integration;
- real TemporalData artifact loading test;
- real ORTHRUS checkpoint inference;
- full_data/e_id compatibility validation;
- end-to-end Kernel SHAP on ORTHRUS;
- final analyst-facing export.

## Current priority

The current priority is to move toward real ORTHRUS inference.

Preferred next steps:

1. audit the current repo state;
2. check whether interrupted sidecar-generator changes created incomplete or inconsistent files;
3. inspect ORTHRUS inference requirements;
4. understand required TemporalData fields, `full_data`, `e_id`, and neighbor loader behavior;
5. integrate model/checkpoint construction around the implemented RealOrthrusAnoAdapter core;
6. test with real ORTHRUS TemporalData artifact;
7. connect Kernel SHAP to ORTHRUS mode.

Avoid implementing extra alternatives until the real ORTHRUS inference path works.

## Important implementation philosophy

Do not implement easier alternative versions just to postpone the final version.

Instead:

- analyze real ORTHRUS requirements carefully;
- implement the minimal final-path component;
- add targeted tests;
- keep dummy mode working only as regression/debug support.

The goal is not to expand the framework horizontally, but to converge toward the final thesis pipeline.
