# Diario implementativo pipeline ORTHRUS

## Obiettivo finale

La pipeline finale deve spiegare con Kernel SHAP uno score prodotto da ORTHRUS-ano su artifact nativi `TemporalData`. Il percorso previsto costruisce un `OrthrusAlertCase`, genera componenti interpretabili, neutralizza le componenti inattive preservando ordine ed edge identity, esegue ORTHRUS e associa i contributi SHAP agli edge e ai metadata leggibili.

## Stato prima di questo step

Il progetto disponeva della pipeline dummy completa, del contenitore ORTHRUS, del builder, delle perturbazioni e dei mapping sintetici. `RealOrthrusAnoAdapter` esponeva soltanto metodi stub: non poteva invocare un modello né ridurre gli edge loss. Anche la modalità ORTHRUS di `pipeline.py` era, e rimane, non implementata.

## Step corrente

È stato implementato il core definitivo di `RealOrthrusAnoAdapter` tramite dependency injection. L'adapter riceve un modello già costruito, un device opzionale, la riduzione `mean` e il controllo opzionale della modalità eval. Non carica checkpoint e non costruisce varianti alternative del modello.

Nel secondo step è stato aggiunto `adapters/orthrus_runtime.py`: un layer di caricamento import-safe che verifica checkout e dipendenze esterne, carica model factory e artifact soltanto a runtime e prepara una singola inferenza reale non perturbata.

L'audit successivo del checkout `external/orthrus` ha permesso di aggiungere una via ufficiale: usa la configurazione ORTHRUS, ricostruisce `full_data` insieme ai tre split, carica il checkpoint completo e prepara un batch con il loader temporale originale.

Per rendere questa via utilizzabile senza scrivere codice ad hoc è stato aggiunto un template JSON con placeholder e un launcher CLI minimale. Il launcher traduce il JSON in `OrthrusOfficialRuntimeConfig`, esegue lo smoke test e stampa il risultato essenziale.

## Scelte architetturali

Il modello viene creato fuori dall'adapter perché configurazione, classi ORTHRUS e checkpoint dipendono dall'ambiente reale. L'adapter resta il confine stabile fra Kernel SHAP e ORTHRUS: accetta un `OrthrusAlertCase` e chiama esattamente `model(batch, full_data, inference=True)`.

ORTHRUS produce una loss per edge, mentre Kernel SHAP richiede una funzione che restituisca un singolo numero per ogni maschera. La riduzione scelta è `mean(edge_losses)`, coerente con la decisione architetturale iniziale e indipendente dal numero assoluto di edge, che la neutralizzazione mantiene comunque stabile.

Dopo l'adapter serve un runtime perché codice ORTHRUS, checkpoint e artifact hanno dipendenze e path propri. La via ufficiale importa pigramente i moduli esterni e segue le funzioni `load_all_datasets`, `build_model`, `load_model` e `batch_loader_factory`. I moduli iniettati sono punti di sostituzione per i test, non una pipeline alternativa.

## Comportamento implementato

L'adapter richiede modello, `temporal_data` e `full_data`; trasferisce batch e contesto al device se supportato; invoca `model.eval()` quando richiesto; usa `torch.no_grad()` quando Torch è disponibile; accetta tensori o sequenze numeriche; valida forma, non-vuotezza e corrispondenza con il numero di edge; restituisce la media come `float` Python. Il conteggio usa `edge_index` con shape `[2, E]`, poi `src` e infine `msg`.

Il runtime generico continua a supportare tre artifact autonomi. La nuova via ufficiale valida checkout, YAML e directory `model_epoch_N`; carica cfg e split, ricostruisce `full_data`, costruisce e ripristina il modello, seleziona grafo e batch, quindi costruisce il caso e usa l'adapter. Lo smoke test restituisce score, cardinalità, dataset, split, indici, device e warning.

## Test aggiunti

Sono stati aggiunti test con modello fake per chiamata e argomenti, media delle loss, `eval`, device, validazione di caso e dati mancanti, output vuoto o non numerico, mismatch della cardinalità, riduzione non supportata e fallback di conteggio. Un test Torch opzionale verifica tensor output e inferenza senza gradienti.

Per il runtime sono stati aggiunti test su import senza ORTHRUS, availability report, errori per checkout e path mancanti, loader failure contestualizzati e percorso completo con loader fake. I test della via ufficiale verificano cfg, caricamento dataset e checkpoint, costruzione modello, factory dei batch, selezione degli indici, risultato ed errori principali. Il percorso attraversa davvero `OrthrusAlertCase` e `RealOrthrusAnoAdapter`.

La CLI è testata senza artifact reali: i test verificano lettura e validazione JSON, costruzione della dataclass, chiamata al runtime sostituita con un fake, riepilogo stampato ed errori chiari per file assente o JSON malformato.

## Cosa non è ancora implementato

Le API ufficiali sono identificate e integrate, ma non sono ancora state eseguite con artifact reali. Restano la verifica del caricamento effettivo di `.TemporalData.simple` e `model_epoch_N`, della coerenza fra stato del neighbor loader, `e_id` e `full_data`, del device reale e infine l'orchestrazione end-to-end in `pipeline.py`.

## Prossimi step previsti

- reperire gli artifact preprocessati e una directory `model_epoch_N` completa;
- eseguire lo smoke test ufficiale prima su CPU;
- verificare `TemporalData`, `full_data`, neighbor loader ed `e_id` durante l'inferenza;
- eseguire un smoke test locale su artifact reale;
- collegare builder, neutralizzazione, adapter ed explainer nella modalità ORTHRUS.

## Cronologia aggiornamenti

- 2026-08-11: implementato il core di `RealOrthrusAnoAdapter` con model injection, chiamata di inferenza, validazione delle loss e riduzione media; aggiunti test fake e Torch opzionale.
- 2026-08-12: aggiunto il runtime ORTHRUS con import dinamico, controlli di disponibilità, caricamento configurabile e smoke test non perturbato; `pipeline.py` resta volutamente stub per Kernel SHAP ORTHRUS.
- 2026-08-12: l'audit del checkout ufficiale ha confermato la firma dell'adapter e l'output per-edge. Il runtime è stato esteso con una via ufficiale basata su `cfg`, `load_all_datasets`, `build_model`, `load_model` e `batch_loader_factory`. `full_data` viene ricostruito dai tre split; il checkpoint `model_epoch_N` ripristina sia `state_dict.pkl` sia `neighbor_loader.pkl`. Lo step copre una sola inferenza non perturbata: restano il test con artifact reali, la verifica temporale/device e Kernel SHAP end-to-end.
- 2026-08-12: aggiunti il template `examples/orthrus_official_smoke_config.example.json` e la CLI `scripts/run_orthrus_official_smoke.py`. Rendono ripetibile il primo smoke test reale senza includere dataset o checkpoint nel repository. La CLI esegue ancora soltanto un batch non perturbato; con gli artifact disponibili restano da validare caricamento, stato temporale, `e_id/full_data` e device prima dell'integrazione Kernel SHAP.
