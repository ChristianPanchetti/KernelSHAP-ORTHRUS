# KSHAP_ORTHRUS

Framework di explainability basato su Kernel SHAP per il modulo di anomaly detection ORTHRUS-ano.

## Stato del progetto

Il repository contiene due percorsi distinti:

- **Dummy/debug:** pipeline end-to-end funzionante basata su `LogDataset`, `DummyOrthrusAnoAdapter` e `shap.KernelExplainer`. Serve per test, sanity check e regressione; non riproduce il comportamento di ORTHRUS-ano.
- **ORTHRUS reale:** integrazione ancora incompleta. Il percorso finale userà artifact ORTHRUS `TemporalData`, non `LogDataset`. Il core di `RealOrthrusAnoAdapter` è implementato per un modello già costruito; caricamento del checkpoint e modalità `orthrus` di `pipeline.py` sono ancora futuri.

Sono già implementati e testati con dati sintetici:

- `OrthrusAlertCase` e il loader di artifact;
- `OrthrusInterpretableBuilder`;
- `OrthrusPerturbationManager`;
- `ArtifactOnlyMappingProvider`, `SidecarMappingProvider` e matching DB offline;
- generazione e validazione dei sidecar.
- runtime configurabile e smoke test non perturbato per modello/artifact esterni.

Non sono ancora disponibili inferenza con checkpoint ORTHRUS reale, validazione con artifact DARPA reali ed esecuzione Kernel SHAP end-to-end in modalità ORTHRUS.

Il modulo `adapters/orthrus_runtime.py` prepara il percorso di inferenza reale senza importare ORTHRUS o Torch all'avvio del progetto. Richiede path locali a checkpoint, `TemporalData` e `full_data`, più una factory del modello esterna. `run_unperturbed_smoke_test(...)` esegue una sola inferenza senza perturbazioni e restituisce score e cardinalità; non è ancora la pipeline SHAP ORTHRUS.

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

La modalità seguente è intenzionalmente non funzionante finché l'adapter reale non sarà implementato:

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

La directory `external/orthrus/` **non è inclusa**. Il codice ORTHRUS-ano, i dataset DARPA, Postgres, gli artifact e i checkpoint devono essere gestiti esternamente e non committati. Dettagli in [docs/ORTHRUS_INTEGRATION.md](docs/ORTHRUS_INTEGRATION.md).

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
