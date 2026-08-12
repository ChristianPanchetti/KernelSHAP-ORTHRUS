# Pipeline dummy e legacy

## Scopo del modulo

La pipeline dummy verifica l'intero ciclo Kernel SHAP senza dipendere da ORTHRUS, Torch, dataset DARPA o checkpoint. È legacy/debug: utile per regressione e comprensione, ma non è una semplificazione valida dell'input reale di ORTHRUS.

## Posizione nella pipeline

Il flusso parte da `logs_input.json`, passa per `LogInputLoader` e `LogDataset`, costruisce componenti con `InterpretableGraphBuilder`, applica maschere con `PerturbationManager`, calcola uno score con `DummyOrthrusAnoAdapter` e usa `KernelSHAPExplainer` e `ResultExporter`.

## Come funziona

`preprocessing/schema.py` definisce `LogRecord` normalizzato e `LogDataset`. `log_input_loader.py` accetta una lista JSON o un oggetto con `records`, normalizza campi come identificativi, tempi, percorso eseguibile e argomenti, conservando anche il payload grezzo.

`InterpretableGraphBuilder` associa ogni record a una componente binaria. Supporta record singoli, executable path, entity id, relazioni parent-child, finestre temporali ed event type. Se `max_components` è superato, conserva i gruppi maggiori e fonde gli altri in `OTHER`. `local_subgraph` resta intenzionalmente non implementato.

`PerturbationManager` interpreta la maschera: `drop_records` elimina i record inattivi; `neutralize_command` mantiene i record ma azzera le informazioni di comando. `drop_time_window` è disponibile soltanto con il grouping coerente. Altre modalità dichiarate sono placeholder espliciti.

Il dummy adapter produce uno score deterministico da numero di record, diversità dei path e parole chiave artificialmente “sospette”, poi applica una sigmoide. Il wrapper SHAP costruisce la funzione maschera → dataset perturbato → score, usa `shap.KernelExplainer`, calcola baseline e input completo e ordina i contributi per valore assoluto. L'export scrive JSON e, opzionalmente, CSV.

## Perché è stato progettato così

Un modello deterministico rende verificabili riproducibilità e additività SHAP. La separazione fra builder, perturbazione e adapter anticipa l'architettura reale senza fingere che `LogDataset` sia un input ORTHRUS.

## Stato attuale

Il percorso è completo e testato via CLI. Rimane supportato come sanity check quando evolveranno i moduli ORTHRUS.

## Limiti e TODO

Lo score non ha significato di sicurezza né corrisponde agli edge loss di ORTHRUS. Le spiegazioni esportate descrivono record/processi della fixture, non provenance graph reali. Non va esteso per sostituire il percorso `TemporalData`.

## File collegati

`logs_input.json`, `preprocessing/schema.py`, `preprocessing/log_input_loader.py`, `perturbation/interpretable_graph_builder.py`, `perturbation/perturbation_manager.py`, `adapters/orthrus_ano_adapter.py`, `xai/kernel_shap_explainer.py` e `xai/result_exporter.py`.
