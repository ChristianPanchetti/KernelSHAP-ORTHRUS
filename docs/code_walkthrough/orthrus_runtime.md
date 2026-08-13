# Runtime e smoke test ORTHRUS

## Scopo del modulo

`adapters/orthrus_runtime.py` prepara una singola inferenza ORTHRUS reale senza introdurre dipendenze obbligatorie all'import del progetto. Il modulo conserva il runtime generico preesistente e aggiunge il percorso ufficiale basato su `cfg`.

## Posizione nella pipeline

Il runtime si trova fra gli artifact ORTHRUS e `RealOrthrusAnoAdapter`: carica dataset e stato, costruisce modello e batch, crea `OrthrusAlertCase` e restituisce uno score. Non genera maschere, non perturba edge e non esegue Kernel SHAP.

## Come funziona

La via ufficiale usa `OrthrusOfficialRuntimeConfig`. Importa pigramente `config.py`, `data_utils.py` e `factory.py` dal checkout, lasciando importabile KSHAP anche senza Torch/PyG/ORTHRUS. In test gli stessi moduli possono essere iniettati come fake.

`load_all_datasets(cfg)` produce train, val, test, `full_data` e il numero massimo di nodi. `graph_index` sceglie una finestra dello split e `batch_index` sceglie un batch prodotto da `batch_loader_factory`. Il modello viene creato da `build_model` e, normalmente, ripristinato con `load_model` dalla directory `model_epoch_N`.

Durante questa operazione la working directory viene impostata temporaneamente alla root ORTHRUS, perché la configurazione ufficiale usa `./artifacts`; un blocco `finally` ripristina sempre la directory iniziale.

`full_data` resta CPU perché gli encoder ufficiali lo interrogano con identificatori `e_id` portati su CPU. Il batch viene invece spostato sul device configurato prima del loader. L'adapter viene quindi usato senza un ulteriore trasferimento automatico di `full_data`.

## Perché è stato progettato così

Il runtime generico presupponeva file separati per batch e `full_data`, mentre ORTHRUS costruisce entrambi attraverso configurazione e dataset loader. La via dedicata rispecchia il flusso ufficiale e rende esplicita l'importanza di `neighbor_loader.pkl`, senza alterare il codice esterno.

## Stato attuale

Configurazione, import lazy, caricamento cfg/dataset, selezione split-grafo-batch, costruzione modello, caricamento checkpoint completo, pesi opzionali e chiamata all'adapter sono implementati. Il risultato include score, cardinalità, dataset, split, indici, device e warning.

Per l'uso manuale, `scripts/run_orthrus_official_smoke.py` legge il template JSON in `examples/`, costruisce `OrthrusOfficialRuntimeConfig`, richiama il runtime e presenta un riepilogo compatto. Gli import del runtime restano lazy, quindi leggere o testare la CLI non richiede Torch/PyG.

## Limiti e TODO

Mancano il primo smoke test con artifact reali, la validazione su GPU, le perturbazioni nel runtime e Kernel SHAP end-to-end. `from_weights_path` non sostituisce lo stato del neighbor loader. Il loader YAML ufficiale non supporta un file arbitrario: `config_path` deve essere il suo `config/orthrus.yml`.

## File collegati

`adapters/orthrus_runtime.py`, `scripts/run_orthrus_official_smoke.py`, `examples/orthrus_official_smoke_config.example.json`, `adapters/orthrus_ano_adapter.py`, `preprocessing/orthrus_alert_case.py`, `external/orthrus/src/data_utils.py`, `external/orthrus/src/factory.py` e `docs/ORTHRUS_INTEGRATION.md`.
