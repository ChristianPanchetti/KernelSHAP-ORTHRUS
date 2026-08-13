# Entrypoint, CLI e pipeline

## Scopo del modulo

Questo gruppo trasforma gli argomenti da riga di comando in configurazione applicativa e avvia il percorso dummy oppure quello ORTHRUS. Mantiene l'orchestrazione separata dalle trasformazioni dei dati e dal modello.

## Posizione nella pipeline

`main.py` è il punto di ingresso. Delega il parsing ad `arg_parser.py`, costruisce le dataclass definite in `config.py`, configura il logging tramite `logger.py` e chiama `run_kernel_shap_pipeline` in `pipeline.py`.

## Come funziona

La CLI espone input, output, modalità, adapter, campionamento SHAP, raggruppamento e perturbazione. `main.py` converte stringhe e numeri in `AppConfig`; `ensure_output_dirs` prepara le directory di destinazione. Le eccezioni vengono registrate e producono codice di uscita 1.

In modalità `dummy`, `pipeline.py` carica `logs_input.json`, costruisce lo spazio interpretabile legacy, crea il gestore delle perturbazioni, seleziona l'adapter, esegue Kernel SHAP ed esporta il risultato. Questo percorso funziona perché tutti i contratti sono interni al repository e usano `LogDataset`.

La modalità `orthrus` controlla soltanto la coerenza fra modalità e tipo di adapter, poi entra in `_run_orthrus_pipeline`, che solleva immediatamente `NotImplementedError`. Non carica ancora `TemporalData`, non costruisce un caso e non collega le perturbazioni ORTHRUS all'explainer.

## Perché è stato progettato così

La diramazione precoce impedisce di passare per errore un `LogDataset` al modello reale o un `OrthrusAlertCase` al dummy adapter. Le dataclass immutabili rendono esplicite le opzioni condivise e riducono configurazioni implicite.

## Stato attuale

Entrypoint, configurazione, logging e pipeline dummy sono operativi. La CLI principale espone ancora opzioni proprie del percorso dummy e non orchestra Kernel SHAP su artifact ORTHRUS. Separatamente, `scripts/run_orthrus_official_smoke.py` costruisce il runtime ufficiale, carica modello, checkpoint e contesto e invoca il `RealOrthrusAnoAdapter` già implementato su un batch non perturbato.

## Limiti e TODO

Occorre ancora collegare alla modalità `orthrus` di `pipeline.py` il runtime validato, la costruzione delle componenti, la neutralizzazione e il contratto dell'explainer. Prima va eseguito lo smoke test con artifact reali. La modalità reale non deve riutilizzare `LogDataset`: la pipeline finale usa `TemporalData`.

## File collegati

`main.py`, `arg_parser.py`, `config.py`, `logger.py`, `pipeline.py`, `adapters/orthrus_ano_adapter.py` e `xai/kernel_shap_explainer.py`.
