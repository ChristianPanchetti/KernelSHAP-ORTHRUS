# Test e copertura del comportamento

## Scopo del modulo

La suite protegge la pipeline dummy e verifica i componenti ORTHRUS senza richiedere servizi, dataset DARPA o checkpoint. I test sintetici fissano i contratti locali prima dell'integrazione reale.

## Posizione nella pipeline

I test seguono i livelli del sistema: normalizzazione e componenti, perturbazioni, adapter, SHAP, CLI, mapping, sidecar e scaffolding ORTHRUS. `tests/conftest.py` fornisce path della fixture, root del progetto e helper CLI/JSON.

## Come funziona

I test dummy principali sono:

- `test_interpretable_graph_builder.py`, che copre grouping, fallback, limite `OTHER`, finestre ed errori dichiarati;
- `test_perturbation_manager.py` e `test_perturbation.py`, che verificano maschere, rimozione, neutralizzazione legacy e modalità non implementate;
- `test_pipeline_smoke.py`, che esegue la CLI e controlla JSON/CSV;
- `test_reproducibility.py` e `test_shap_additivity.py`, che controllano stabilità col seed e coerenza fra baseline, score e contributi;
- il test dummy presente anche nei file ORTHRUS, che protegge la compatibilità legacy.

I test ORTHRUS sintetici sono:

- `test_orthrus_scaffolding.py`, che controlla import senza Torch, separazione dei tipi e fallimento esplicito di adapter/pipeline stub;
- `test_orthrus_perturbation.py`, che usa `FakeTemporalData` per verificare drop, ricostruzione di `edge_index` e neutralizzazione;
- `test_orthrus_mapping.py` e `test_orthrus_mapping_sidecar.py`, che coprono artifact-only, descrizioni, loader pickle, sidecar e join DB in memoria;
- `test_orthrus_sidecar_generator.py`, che copre generazione, caricamento, validazione, fallback e warning.

Il test che conserva dtype e device usa `pytest.importorskip("torch")`. Se Torch non è installato viene saltato, perché Torch è una dipendenza opzionale e non appartiene ai requirements base.

## Perché è stato progettato così

Oggetti sintetici rendono i test rapidi, riproducibili e utilizzabili in qualunque ambiente. Il fallimento esplicito della modalità ORTHRUS è oggi comportamento atteso: evita di scambiare uno scaffold per inferenza reale.

## Stato attuale

La suite copre bene la pipeline dummy e i contratti strutturali ORTHRUS. Non misura la correttezza scientifica dello score dummy né la compatibilità con un checkpoint.

## Limiti e TODO

Servono test del `RealOrthrusAnoAdapter` con modello fake e loss note, validazione di `mean(edge_losses)`, casi senza edge, propagazione di device/errori e compatibilità `full_data/e_id`. In seguito serviranno smoke test locali, separati e opzionali, con `TemporalData`, configurazione e checkpoint reali, senza inserirli nella suite portabile o nel repository.

## File collegati

Tutta la cartella `tests/`, `requirements-dev.txt`, `logs_input.json` e i moduli descritti negli altri capitoli del walkthrough.
