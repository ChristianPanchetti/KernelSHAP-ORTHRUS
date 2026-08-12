# Walkthrough del codice

## Scopo del modulo

Questa cartella è una guida discorsiva all'architettura di KSHAP_ORTHRUS. Serve a ricostruire rapidamente il percorso dei dati, le responsabilità dei moduli e i confini fra ciò che è già funzionante e ciò che richiede ancora l'integrazione con ORTHRUS reale.

## Posizione nella pipeline

Il progetto contiene due pipeline. Quella dummy, basata su `LogDataset`, è completa ma ha finalità legacy/debug. Quella finale dovrà lavorare direttamente con `TemporalData`, costruire un `OrthrusAlertCase`, applicare maschere tramite neutralizzazione e interrogare `RealOrthrusAnoAdapter`. Quest'ultimo percorso non è ancora end-to-end.

## Come funziona

La lettura consigliata segue il flusso dei dati:

1. [Entrypoint e orchestrazione](entrypoints_and_pipeline.md)
2. [Pipeline dummy](dummy_pipeline.md)
3. [Caso ORTHRUS e caricamento](orthrus_case_and_loading.md)
4. [Componenti interpretabili ORTHRUS](orthrus_interpretable_components.md)
5. [Perturbazioni ORTHRUS](orthrus_perturbations.md)
6. [Mapping e sidecar](mapping_and_sidecars.md)
7. [Adapter del modello](adapters.md)
8. [Runtime e smoke test ORTHRUS](orthrus_runtime.md)
9. [Strategia di test](tests.md)

## Perché è stato progettato così

La separazione evita di confondere la rappresentazione semplificata usata per verificare Kernel SHAP con il contratto dati reale di ORTHRUS. Il mapping degli eventi è inoltre separato dall'inferenza: `full_data` è contesto numerico del modello, mentre `event_uuid` e i sidecar servono a rendere leggibile la spiegazione.

## Stato attuale

La pipeline dummy e i componenti ORTHRUS strutturali sono coperti da test. La suite corrente non richiede DARPA, Postgres o checkpoint. `RealOrthrusAnoAdapter` e l'orchestrazione ORTHRUS restano incompleti.

## Limiti e TODO

La documentazione descrive il codice corrente, inclusi alcuni docstring sorgente ormai datati. Dovrà essere aggiornata quando saranno definiti il checkpoint, il caricamento di `full_data`, il replay temporale e il contratto esatto degli edge loss.

## File collegati

I riferimenti generali sono `README.md`, `AGENTS.md`, `PROJECT_CONTEXT.md` e `docs/ORTHRUS_INTEGRATION.md`.
