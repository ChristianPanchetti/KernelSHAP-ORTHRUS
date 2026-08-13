# Integrazione ORTHRUS reale

`external/orthrus/` contiene il checkout ufficiale. Il contratto verificato è `model(batch, full_data, inference=True)` con un tensore monodimensionale di loss, una per edge. Il core di `RealOrthrusAnoAdapter` riduce queste loss con la media.

## Percorso ufficiale

`OrthrusOfficialRuntimeConfig` e `run_official_orthrus_smoke_test` seguono il flusso della repository esterna:

1. le funzioni ufficiali `get_runtime_required_args` e `get_yml_cfg` producono `cfg`;
2. `load_all_datasets(cfg)` carica train/val/test e costruisce `full_data` concatenandoli;
3. viene scelta una finestra con `split` e `graph_index`;
4. `build_model(data_sample, device, cfg, max_node_num)` costruisce il modello;
5. `load_model(model, model_epoch_dir)` carica `state_dict.pkl` e `neighbor_loader.pkl`;
6. `batch_loader_factory` crea i batch e `batch_index` ne seleziona uno;
7. il runtime costruisce `OrthrusAlertCase` e usa l'adapter per ottenere lo score.

La configurazione richiede `external_root`, il `config_path` ufficiale, `dataset_name`, `model_epoch_dir`, split, indici e device. È disponibile anche `from_weights_path`: i pesi vengono applicati dopo il checkpoint completo, quindi sostituiscono lo state dict ma non lo stato temporale. Per default il checkpoint completo è obbligatorio, perché i soli `weights/<DATASET>.pkl` non contengono `neighbor_loader.pkl`.

Il loader ufficiale sceglie sempre `external_root/config/orthrus.yml`; per questo `config_path` deve indicare esattamente quel file. Gli override accettano una mappa di chiavi dotted oppure una sequenza di argomenti CLI ORTHRUS.

Esempio concettuale:

```python
from pathlib import Path
from adapters.orthrus_runtime import (
    OrthrusOfficialRuntimeConfig,
    run_official_orthrus_smoke_test,
)

cfg = OrthrusOfficialRuntimeConfig(
    external_root=Path("external/orthrus"),
    config_path=Path("external/orthrus/config/orthrus.yml"),
    dataset_name="THEIA_E3",
    model_epoch_dir=Path("artifacts/.../trained_models/model_epoch_1"),
    split="test",
    graph_index=0,
    batch_index=0,
    device="cpu",
)
result = run_official_orthrus_smoke_test(cfg)
```

I dataset `.TemporalData.simple` devono trovarsi nei percorsi calcolati dalla configurazione ORTHRUS. `full_data` non è richiesto come file autonomo: viene ricostruito da `load_all_datasets`. Il grafo selezionato viene spostato sul device prima del batching; `full_data` resta sul device originale, coerentemente con il codice ORTHRUS che lo indicizza tramite `e_id.cpu()`.

Poiché ORTHRUS definisce la radice degli artifact come `./artifacts`, la funzione esegue il percorso ufficiale con working directory temporaneamente impostata a `external_root` e ripristina sempre quella del chiamante.

## Limiti attuali

Lo smoke test è non perturbato e non esegue Kernel SHAP. Non implementa replay manuale del `LastNeighborLoader`, mapping reale, Postgres o la modalità ORTHRUS di `pipeline.py`. Prima del test reale servono artifact preprocessati compatibili e una directory `model_epoch_N` completa; dataset, checkpoint, sidecar e output non vanno committati.

Il precedente `OrthrusRuntimeConfig` resta disponibile per compatibilità, ma rappresenta un percorso generico basato su artifact autonomi e non il contratto ufficiale ORTHRUS.

## Avvio da configurazione JSON

Il file `examples/orthrus_official_smoke_config.example.json` è un modello versionabile privo di path personali e dati sensibili. Va copiato in un file locale e compilato con dataset e directory del checkpoint reali. `from_weights_path` può restare `null`; `overrides` può restare vuoto oppure contenere chiavi dotted accettate dalla CLI ORTHRUS.

Lo smoke test si avvia dalla root del progetto con:

```bash
python scripts/run_orthrus_official_smoke.py path/to/smoke_config.json
```

Lo script stampa score, numero di edge e loss, dataset, split, indice del batch, device e warning. Errori di JSON, path, dipendenze o artifact vengono riportati con un messaggio contestualizzato. Il file di configurazione reale può contenere path infrastrutturali e non va necessariamente committato; il template `.example.json` è invece sicuro da mantenere nel repository.
