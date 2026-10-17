# KSHAP_ORTHRUS

Framework di explainability basato su Kernel SHAP per il modulo di anomaly detection ORTHRUS-ano.

## Stato del progetto

Il repository contiene due percorsi distinti:

- **Dummy/debug:** pipeline end-to-end funzionante basata su `LogDataset`, `DummyOrthrusAnoAdapter` e `shap.KernelExplainer`. Serve per test, sanity check e regressione; non riproduce il comportamento di ORTHRUS-ano.
- **ORTHRUS reale:** integrazione ancora incompleta. Il percorso finale usa artifact ORTHRUS `TemporalData`, non `LogDataset`. Il core di `RealOrthrusAnoAdapter`, il runtime generico, il runtime ufficiale cfg-driven e la CLI dello smoke test sono implementati. La modalità `orthrus` Kernel SHAP di `pipeline.py` resta futura.

Sono già implementati e testati con dati sintetici:

- `OrthrusAlertCase` e il loader di artifact;
- `OrthrusInterpretableBuilder`;
- `OrthrusPerturbationManager`;
- `ArtifactOnlyMappingProvider`, `SidecarMappingProvider` e matching DB offline;
- generazione e validazione dei sidecar.
- runtime generico e runtime ufficiale cfg-driven con smoke test non perturbato;
- caricamento ufficiale di directory `model_epoch_N` complete di `state_dict.pkl` e `neighbor_loader.pkl`;
- CLI e configurazione di esempio per avviare lo smoke test ufficiale.

Il percorso di inferenza è implementato, ma non è ancora stato eseguito con checkpoint e artifact ORTHRUS reali. Restano inoltre la validazione con artifact DARPA reali e Kernel SHAP end-to-end in modalità ORTHRUS.

Il modulo `adapters/orthrus_runtime.py` prepara il percorso di inferenza reale senza importare ORTHRUS o Torch all'avvio del progetto. Richiede path locali a checkpoint, `TemporalData` e `full_data`, più una factory del modello esterna. `run_unperturbed_smoke_test(...)` esegue una sola inferenza senza perturbazioni e restituisce score e cardinalità; non è ancora la pipeline SHAP ORTHRUS.

Il percorso aderente alla repository ufficiale si può avviare copiando e compilando `examples/orthrus_official_smoke_config.example.json`, quindi eseguendo:

```bash
python scripts/run_orthrus_official_smoke.py path/to/smoke_config.json
```

Servono gli artifact `.TemporalData.simple` nei path calcolati da ORTHRUS e una directory `model_epoch_N` contenente `state_dict.pkl` e `neighbor_loader.pkl`. Dataset, checkpoint e artifact DARPA sono locali e non devono essere committati. Il comando esegue un solo batch non perturbato: non avvia ancora Kernel SHAP.

## Pipeline

Pipeline dummy:

```text
logs_input.json -> LogDataset -> componenti interpretabili -> perturbazioni
-> DummyOrthrusAnoAdapter -> Kernel SHAP -> output JSON/CSV
```

Pipeline ORTHRUS prevista:

```text
TemporalData + full_data -> OrthrusAlertCase -> componenti interpretabili
-> maschere Kernel SHAP -> neutralizzazione -> RealOrthrusAnoAdapter
-> edge losses -> mean(edge_losses) -> SHAP values
```

Per ORTHRUS reale la strategia preferita è `neutralize_edges` (alias `mask_edge_features`): mantiene numero e ordine degli edge e quindi evita di alterare gli identificativi globali prodotti dall'inserimento cronologico nel `LastNeighborLoader`. `drop_edges` rimane disponibile per test sintetici, debug o modelli non stateful; non è considerato sicuro per ORTHRUS reale senza gestione esplicita di `full_data` ed `e_id`.

## Installazione e test

Dipendenze runtime minime:

```bash
python -m pip install -r requirements.txt
```

Dipendenze di sviluppo e test:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

PyTorch, PyTorch Geometric e ORTHRUS reale non sono dipendenze base: saranno dipendenze opzionali dell'integrazione reale.

## Esecuzione dummy

`logs_input.json` è una fixture di esempio anonimizzata, mantenuta in root perché usata dai test e dai comandi di avvio correnti.

```bash
python main.py --mode dummy --adapter dummy \
  --input logs_input.json \
  --output outputs/kernel_shap_results.json
```

È possibile aggiungere `--output-csv outputs/kernel_shap_ranking.csv`. La directory `outputs/` è mantenuta nella struttura, ma i risultati e i log generati sono ignorati da Git.

La modalità seguente è intenzionalmente non funzionante perché l'orchestrazione Kernel SHAP ORTHRUS in `pipeline.py` non è ancora implementata; l'adapter e lo smoke test ufficiale sono invece disponibili separatamente:

```bash
python main.py --mode orthrus --adapter real --input artifact.pt --output outputs/result.json
```

## Mapping e sidecar

I componenti SHAP sono associati agli edge tramite `component_to_edges`. Il mapping artifact-only fornisce informazioni presenti nel `TemporalData`; un sidecar può aggiungere `event_uuid` e metadata originari. Il formato principale è:

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

I sidecar generati sono ignorati per default e non sono necessari per l'inferenza: servono alla leggibilità delle spiegazioni.

## Dipendenza ORTHRUS esterna

La directory `external/orthrus/` può essere presente localmente, ed è attualmente usata dal runtime ufficiale, ma è ignorata da Git e non fa parte del contenuto versionato. Dataset DARPA, Postgres, artifact e checkpoint restano locali e non devono essere committati. Dettagli in [docs/ORTHRUS_INTEGRATION.md](docs/ORTHRUS_INTEGRATION.md).

## Struttura essenziale

- `adapters/`: interfaccia, adapter dummy e core dell'adapter reale con model injection;
- `adapters/orthrus_runtime.py`: loading dinamico e smoke test ORTHRUS non perturbato;
- `preprocessing/`: schema dummy, `OrthrusAlertCase`, loader e mapping;
- `perturbation/`: builder e strategie di perturbazione dummy/ORTHRUS;
- `xai/`: Kernel SHAP ed esportazione;
- `tests/`: test sintetici, smoke test e regressioni;
- `outputs/`: destinazione locale degli output, non tracciati;
- `PROJECT_CONTEXT.md`: contesto architetturale e stato corrente.

## Licenza

MIT, vedere [LICENSE](LICENSE).
# Offline explanation visualizations

From the project directory, regenerate all nine visualization artifacts from an existing result:

```bash
python main.py --plot-json outputs/theia_e5_phase11_final.json
```

This reads only the JSON and does not run ORTHRUS, SHAP computation, CUDA, or PostgreSQL. Alongside the unchanged SHAP bar and waterfall PNG/PDF files, it creates `*_provenance_subgraph.png/.pdf` and `*_summary_table.csv/.png/.pdf`.

The provenance subgraph contextualizes SHAP-attributed components using mapped graph entities and event relations. SHAP values are assigned to interpretable components only; displayed events and counterpart nodes provide semantic context and do not receive independent SHAP attribution. Positive values increase the anomaly score relative to the chosen baseline; negative values decrease it. OTHER remains one aggregate component.

The deterministic graph displays the top five components and up to two counterparts per component, ranked by assigned event frequency. Neutral context copies are labelled by node ID; arrow direction follows recorded source/destination IDs and line width reflects event counts. Omitted counterpart event counts are shown. The table includes all components, top three operations and top two counterpart entities; CSV retains full entity labels and SHAP precision. Missing mapping produces an ID-only graph with an explicit unavailable-context label. Netflow identities use node ID/type without endpoint claims.



## Temporal test-set scan (server only)

Scan mode runs ORTHRUS only, without SHAP, perturbations, PostgreSQL mapping,
DB enrichment or visualization. It reuses the real runtime and verifies the
training checkpoint history, processes all validation batches, then traverses
all test graphs/batches once in official order. Temporal state advances once
per forward and is never restored between scan batches. The single-case
`split`, `graph_index` and `batch_index` in the runtime JSON are ignored for scanning.
Batch anomaly score is `sum(per_edge_losses) / len(per_edge_losses)`, exactly
matching `RealOrthrusAnoAdapter` (including its Python-float aggregation).

The local official ORTHRUS THEIA_E5 configuration points to
`E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv`.
`detection.evaluation_utils.get_ground_truth_nids` delegates to
`labelling.get_ground_truth(cfg)`: CSV UUIDs are resolved to integer node IDs
using the official database UUID lookup. No ready offline UUID/node-ID ground
truth artifact is bundled here. To keep scans database-free, provide a previously
exported official label set from the same dataset/artifact build. The scanner
never calls that database-backed function itself. An export must have this schema:

```json
{
  "dataset": "THEIA_E5",
  "ground_truth_level": "node",
  "official_label_function": "labelling.get_ground_truth",
  "ground_truth_source": "Describe the actual official CSV and UUID lookup export source",
  "ground_truth_version": "darpa",
  "ground_truth_relative_path": ["E5-THEIA/node_THEIA_1_Firefox_Drakon_APT_BinFmt_Elevate_Inject.csv"],
  "node_ids": []
}
```

The exporter below fills `node_ids` with the complete sorted integer set returned
as the first item by official `labelling.get_ground_truth(cfg)`. The empty list
above is only a schema example, not THEIA_E5 ground truth. Preserve its version
and configured CSV paths; scanner checks both against the official configuration. This export is
an explicit prerequisite: do not substitute UUIDs, invent labels or silently
use an empty set when labels are unavailable. The scanner records the export
path, SHA-256, configured label files and declared source. The server must verify
that the export uses the same UUID/node-ID namespace as the temporal artifacts;
these metadata checks cannot prove that correspondence.

Export once from `KSHAP_ORTHRUS` on the real server/container:

```bash
python scripts/export_scan_ground_truth.py \
  --orthrus-config examples/orthrus_theia_e5_real_smoke.json \
  --output outputs/theia_e5_official_node_ground_truth.json
```

This utility loads the official configuration using the same runtime JSON and
overrides, then calls `labelling.get_ground_truth(cfg)` directly (the helper used
by `detection.evaluation_utils.get_ground_truth_nids`). Its database connection
uses the official `cfg.database` settings and official dataset/database selection;
no separate connection settings or credentials are introduced. Run it with the
same server configuration/environment as the real ORTHRUS artifact build.
It does not load a checkpoint, construct a model, run inference/SHAP, or enrich
explanations. Official modules still require the server's ORTHRUS dependencies.
Missing dataset/configuration/label files and failed official DB/UUID lookups
produce an error and nonzero exit status.

The JSON also records `malicious_node_count` and resolved `ground_truth_files`.
IDs remain exact integers, deduplicated and sorted; metadata reports node-level
labels and their official source. No expected count can be established from the
bundled source alone: the official label CSV and live UUID/node-ID mapping must
be verified on the server. The utility prints the exported count on success.

Run from `KSHAP_ORTHRUS` inside the real server/container, with its existing real
runtime JSON and the verified offline official ground-truth export:

```bash
python main.py --mode orthrus --adapter real \
  --orthrus-config examples/orthrus_theia_e5_real_smoke.json \
  --scan-test --scan-output outputs/theia_e5_test_scan.csv \
  --scan-ground-truth outputs/theia_e5_official_node_ground_truth.json
```

The CSV contains one row per test batch, stable graph/batch/global offset IDs,
edge count, and mean/min/median/p95/max edge-loss statistics. Counts of unique
malicious nodes in the union of endpoints, unique source nodes and unique
destination nodes, plus `contains_malicious_node`, are **derived descriptive
batch fields** based on official node membership. They are not official batch
labels, temporally localized attack-event labels, or malicious-edge labels.
No edge-label field is created. The summary records that distinction explicitly.

`outputs/theia_e5_test_scan_summary.json` contains aggregate batch statistics,
percentiles, descriptive groups by node membership, and deterministic suggested
cases for later manual selection of 3–5 SHAP studies. Median representatives
minimize distance to the relevant group's median; high/low candidates use extrema;
ties use official scan order. Missing categories are marked unavailable and
categories may suggest the same batch. No SHAP runs automatically and no candidate
is assigned TP/FP/FN terminology.

Official evaluation metrics are recorded as not computed. The official evaluator
requires validation/test time-window edge-loss CSVs, configured node score
aggregation and validation thresholds (and potentially database-backed label/
window logic); those artifacts are not emitted by this compact scan. Batch means
cannot replace official detector metrics. Use the existing official evaluation
workflow separately if those metrics are needed.

Each CSV row is flushed immediately. A successful summary is written only after
the sequence completes; an old summary is removed when a new CSV scan starts.
An interrupted run leaves partial CSV data and requires restarting from the
verified checkpoint. There is no resume from CSV. Only batch statistics are kept
for the summary; per-edge losses are discarded after each batch. The existing
runtime still loads its normal temporal artifacts/full history.

No real scan is required for local testing. On the server verify checkpoint and
artifact paths, complete offline label provenance/namespace, device behavior,
and the first `test/0/0` mean (approximately `0.7171946` for the validated artifacts;
this value is not hard-coded).
