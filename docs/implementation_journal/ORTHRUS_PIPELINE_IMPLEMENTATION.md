# Diario implementativo pipeline ORTHRUS

## Obiettivo finale

La pipeline finale deve spiegare con Kernel SHAP uno score prodotto da ORTHRUS-ano su artifact nativi `TemporalData`. Il percorso previsto costruisce un `OrthrusAlertCase`, genera componenti interpretabili, neutralizza le componenti inattive preservando ordine ed edge identity, esegue ORTHRUS e associa i contributi SHAP agli edge e ai metadata leggibili.

## Stato prima di questo step

Il progetto disponeva della pipeline dummy completa, del contenitore ORTHRUS, del builder, delle perturbazioni e dei mapping sintetici. `RealOrthrusAnoAdapter` esponeva soltanto metodi stub: non poteva invocare un modello né ridurre gli edge loss. Anche la modalità ORTHRUS di `pipeline.py` era, e rimane, non implementata.

## Step corrente

### Preparazione temporale ufficiale per la validazione Fase 9 (2026-09-20)

Audit completato su data_utils.py, temporal.py, encoders.py, factory.py,
orthrus_gnn_training.py e orthrus_gnn_testing.py del checkout ufficiale.
Il training resetta il neighbor loader all'inizio di ogni epoca e salva
neighbor_loader.pkl dopo tutti i grafi train. save_model salva pesi e loader,
non le cache GraphReindexer. Il testing costruisce un modello nuovo (cache
None), carica quel checkpoint, applica gli eventuali pesi pretrained senza
sostituire la memoria e percorre val poi test, senza reset intermedi.
Le cache inizialmente vuote sono quindi il comportamento ufficiale: non va
aggiunto un replay del train attraverso il modello per popolarle.

Il vecchio smoke selezionava direttamente il batch richiesto dopo load_model:
il successo del forward non provava l'allineamento con full_data. Il nuovo
percorso è opzionale (`perturbative=True`, CLI `--phase9`), riutilizza cfg,
dataset, modello e caricamento checkpoint/pesi esistenti. Senza l'opzione il
comportamento dello smoke rimane invariato.

Prima di qualunque forward, la modalità Fase 9 richiede un checkpoint di
training completo, split val/test, _test_mode disattivato, cache None e
cur_e_id == N_train. Non si accontenta del contatore: ricostruisce su un loader
CPU separato la sola topologia del train con insert e gli stessi confini dei
batch ufficiali, e confronta e_id e neighbors degli slot validi con il loader
caricato. Non modifica il checkpoint, non addestra e non esegue forward train.
Se la memoria non coincide si ferma; non corregge artificialmente cur_e_id e
non tenta di interpretare un checkpoint già avanzato come checkpoint train.
Questo confronto verifica la memoria rispetto agli artifact caricati; non
costituisce una certificazione della provenienza dei pesi.

Dopo questa verifica, trasferisce i tensori del loader al device richiesto
preservandone i valori e avanza con inferenza non perturbata lungo il prefisso
ufficiale. Per test/0/0 il prefisso è tutta la validation. Per val/0/0 è vuoto.
Per batch successivi include anche i grafi/batch precedenti dello split.
Controlla contatore e corrispondenza di t, edge_type e msg con le righe globali
di full_data prima di ogni batch. Acquisisce lo snapshot della Fase 9 soltanto
quando raggiunge il batch selezionato, ancora mai passato al modello.

Caso mantenuto: **test/0/0**, atteso 1024 edge, configurazione server con
device=cuda. Nessun passaggio automatico a validation. Il costo reale del
prefisso non è stato misurato; il controllo della topologia richiede una
scansione del train e memoria per un loader CPU aggiuntivo. Se necessario,
val/0/0 può essere selezionato esplicitamente nel JSON come caso diverso:
evita il forward della validation, ma mantiene il controllo del checkpoint.

Il runner costruisce componenti node-based per source (massimo 8, eventuale
OTHER del builder esistente), usa l'ordine del mapping e disattiva la prima
componente in B. Esegue esattamente A1, B, A2, Z dopo il prefisso, con lo stesso
adapter isolato e senza un quinto forward di baseline. Un hook conta le loss
effettive. Per ogni valutazione verifica score finito, 1024 loss, struttura e
target invariati, immutabilità dell'originale e di full_data e ripristino dello
stato. Digest SHA-256 a blocchi evitano di duplicare full_data per i controlli.
A1/A2 sono confrontati con rtol=1e-6 e atol=1e-8; non viene imposta alcuna
direzione agli score B/Z. Il report stampa risultati e offset temporale.

Ambiente: examples/orthrus_theia_e5_real_smoke.json non è stato trovato nella
copia locale, né sono stati trovati artifact reali accessibili. I file
neighbor_loader.pkl sotto .pytest_* sono fixture sintetiche. Torch non è
installato nel Python corrente e il daemon Docker locale non è raggiungibile.
Questo NON implica che configurazione, GPU o artifact non esistano nel
container ORTHRUS sul server Linux, dove è stato eseguito lo smoke precedente.
Non sono disponibili qui i path effettivi del JSON server per verificarli.

**Test reale non eseguito.** A1, B, A2, Z, edge_loss_count reali e ripetibilità
CUDA restano non misurati. La Fase 9 reale non viene dichiarata validata.
La vecchia baseline 0.7162450345895195 non è automaticamente confrontabile:
manca evidenza del suo stato pre-batch. Il runner registra A1 come baseline
preparata e segnala che la comparabilità storica non è dimostrata.

Modifiche di questo step: adapters/orthrus_runtime.py,
scripts/run_orthrus_official_smoke.py, tests/test_orthrus_official_runtime.py,
tests/test_orthrus_official_smoke_script.py e questo diario. Perturbazione,
snapshot/restore, external/orthrus e core Kernel SHAP non sono stati riscritti.

Test mirati eseguiti:

```text
python -m pytest -q tests/test_orthrus_official_runtime.py tests/test_orthrus_official_smoke_script.py --basetemp=.pytest_phase9_temporal_final_tmp
```

Esito: **17 passed**. I controlli sintetici coprono rifiuto del checkpoint con
contatore o vicini errati prima del forward, prefisso vuoto per val/0/0,
ordine validation/test per un batch successivo, batch target non consumato,
esattamente quattro score isolati e opzione CLI. Non sono prove di inferenza
Torch/CUDA; nessuna suite completa è stata eseguita.

Comando nel container Linux, dalla root KSHAP_ORTHRUS contenente la
configurazione reale e dopo avervi trasferito i file aggiornati:

```bash
python scripts/run_orthrus_official_smoke.py examples/orthrus_theia_e5_real_smoke.json --phase9
```

Il JSON server deve indicare THEIA_E5, split=test, graph_index=0, batch_index=0,
device=cuda e gli stessi path di checkout, config/orthrus.yml, model_epoch_dir,
eventuale from_weights_path e overrides dello smoke originale. I path relativi
sono risolti dalla directory di lancio prima del cambio directory verso ORTHRUS;
gli artifact derivano dalla cfg ufficiale. Non creare path fittizi al posto di
quelli server. Se un controllo fallisce, conservare l'errore e non azzerare la
memoria per aggirarlo. Il prossimo passo è raccogliere il report reale di
questo comando; soltanto dopo il suo successo valutare il collegamento SHAP.

### Verifica preliminare del test reale Fase 9 (2026-09-20)

Esito: **test non eseguito; Fase 9 non ancora validata su THEIA_E5 reale**.
La verifica preliminare per split=test, graph_index=0, batch_index=0,
device=cuda e 1024 edge non consente di dimostrare la coerenza temporale
dello stato iniziale. Nessun forward A1/B/A2/Z è stato eseguito, nessuna
componentizzazione è stata costruita e nessuno snapshot è stato acquisito.

Il runtime attuale carica il checkpoint e seleziona direttamente il batch
richiesto, senza attraversare la validation né verificare l'offset del loader.
full_data concatena train, validation e test: per il primo batch di test
l'offset atteso è N_train + N_val. Un checkpoint prodotto dal training
ufficiale contiene invece il loader alla fine del train; il testing ufficiale
esegue la validation prima del test. Inoltre load_model non ripristina le
cache del GraphReindexer. Non è stato possibile verificare se il checkpoint
effettivamente usato nello smoke precedente fosse stato preparato diversamente:
nei percorsi locali controllati non sono stati reperiti la configurazione
concreta dello smoke, gli artifact TemporalData o neighbor_loader.pkl;
è presente soltanto il template di configurazione con placeholder.

Come richiesto, nessun avanzamento temporale, reset o aggiustamento del
contatore è stato applicato per aggirare questa precondizione. Score A1/B/A2/Z,
edge_loss_count, ripetibilità, confronto con 0.7162450345895195 e ripristino
dello stato dopo inferenza rimangono non misurati. Nessun batch o full_data
reale è stato caricato o modificato. Nessun test aggiuntivo, modifica al codice
o collegamento Kernel SHAP è stato effettuato. Per riprendere occorrono la
configurazione e il checkpoint effettivi dello smoke e la verifica del loro
stato pre-batch, inclusa la provenienza del contesto temporale.

### Fase 9 — perturbazione deterministica del current batch (2026-09-17)

Il contesto operativo comunicato dall'utente supera lo stato storico riportato
nelle sezioni successive: gli artifact THEIA_E5 e lo smoke ufficiale sono già
validati (1024 edge, 1024 loss, score originale 0.7162450345895195).
Questo step implementa la perturbazione e l'isolamento delle valutazioni; non
esegue nuovamente il test reale e non collega Kernel SHAP end-to-end.

`OrthrusPerturbationManager.neutralize_edges` (alias `mask_edge_features`)
parte sempre dal caso originale e azzera esclusivamente `x_src` e `x_dst`.
`src`, `dst`, `t`, `edge_index`, cardinalità e ordine, `edge_type` (target),
`msg`, `edge_feats` e `full_data` rimangono invariati. Le feature modificate
sono copiate preservando dtype/device; feature richieste mancanti, dimensioni
incoerenti e indici inattivi fuori batch producono errori espliciti.
`drop_edges` rimane disponibile solo per gli usi legacy/debug.

La semantica per i nodi ripetuti è **tutte le occorrenze nello stesso ruolo**:
un edge inattivo neutralizza tutte le righe source con il suo source e tutte
le righe destination con il suo destination. La propagazione non attraversa
i ruoli e non è ricorsiva. Può quindi neutralizzare feature in componenti
attive. I metadata distinguono gli edge selezionati dalla mask dalle righe
source/destination effettivamente neutralizzate. Questa perturbazione spiega
feature correnti dei nodi a storia, topologia e target fissi; non rimuove eventi.
Non usa gli e_id storici per modificare full_data.edge_type.

L'ordine canonico delle componenti è l'ordine di inserimento del dizionario
`component_to_edges`, condiviso dal manager e da `suggested_component_ids`.
Non ricostruire o riordinare quel mapping tra valutazioni.

`RealOrthrusAnoAdapter(..., isolate_state=True)` acquisisce alla costruzione
uno snapshot in memoria dello stato pre-batch. Prima di ogni score lo ripristina,
poi lo ripristina nuovamente in un `finally`, anche se inferenza o validazione
delle loss falliscono. Include `LastNeighborLoader.cur_e_id`, `neighbors`,
`e_id`, `_assoc` se presente, le cache `GraphReindexer.x_src_cache` e
`x_dst_cache` (compreso `None`) e le associazioni temporanee disponibili.
Ogni ripristino clona i valori: le mutazioni del modello non contaminano lo
snapshot. I pesi non vengono copiati. La modalità isolata richiede eval ed
esclude esplicitamente modelli contrastivi non coperti da questo contratto.
La modalità sequenziale preesistente resta il default.

L'adapter non trasferisce più `full_data` al device: il contesto condiviso resta
intatto, coerentemente con l'indicizzazione CPU dell'encoder ufficiale. Quando
richiesto, il trasferimento del batch avviene su una copia del contenitore.
Il device dei pesi e del loader deve essere già predisposto prima dello snapshot.

Percorso minimo disponibile, senza SHAP (modello già caricato e stato pre-batch
già verificato):

```python
builder = OrthrusInterpretableBuilder(grouping_mode="edge_type")
builder.build(case)
component_ids = builder.suggested_component_ids(case)
perturbation = OrthrusPerturbationManager(case, mode="neutralize_edges")
adapter = RealOrthrusAnoAdapter(model, device=None, isolate_state=True)
score = adapter.score_mask(perturbation, [1] * len(component_ids))
```

`score_mask` rifiuta sia l'assenza di isolamento sia perturbazioni diverse da
`neutralize_edges`. Restituisce la media delle loss del batch completo.
Riutilizzare lo stesso adapter e manager per tutte le mask; non costruire un
nuovo snapshot dopo aver consumato il batch. L'uso è seriale: non condividere
il modello con valutazioni concorrenti.

Verifica eseguita con:

```text
python -m pytest -q tests/test_orthrus_perturbation.py tests/test_real_orthrus_adapter.py tests/test_orthrus_scaffolding.py tests/test_orthrus_official_runtime.py tests/test_orthrus_runtime.py --basetemp=.pytest_phase9_verify_tmp
```

Risultato: **36 passed, 4 skipped**. I test esistenti di neutralizzazione sono
stati aggiornati al nuovo contratto. Sono coperti nodi ripetuti, immutabilità,
ordine non alfabetico, mask successive dall'originale, score A-B-A e ripristino
dopo eccezione con cache sia vuote sia popolate. I quattro skip riguardano Torch
non disponibile localmente (dtype/device, no_grad e due varianti dello stato).
Le varianti NumPy dello stato sono state eseguite. Il test del runtime generico
è aggiornato per verificare la copia del batch e la storia non trasferita.
Nessuna suite completa, inferenza reale o esecuzione SHAP in questo step.

Prossimo singolo test reale THEIA_E5:

1. Nell'ambiente Torch/ORTHRUS, rieseguire i due file di test relativi a
   perturbazione e adapter, senza skip Torch.
2. Preparare lo stesso modello/caso dello smoke validato, fermandosi prima del
   forward. Verificare che il loader sia nello stato immediatamente precedente
   al batch: la selezione di un batch nel runtime attuale non esegue il prefisso
   temporale. Se necessario, avanzare una volta sui batch precedenti prima dello
   snapshot; non presumere che i soli pesi ricostruiscano la storia.
3. Costruire componenti, manager e adapter isolato come sopra. Eseguire A (tutti
   uno), B (una componente inattiva), A, e la mask tutti zero. Verificare 1024
   edge/loss, score finiti, ripetibilità di A e invarianza di struttura, target,
   storia e stato dopo ciascuna chiamata. Un cambiamento dello score non è
   matematicamente garantito per qualsiasi componente: verificare anche le
   feature effettivamente perturbate.
4. Con le stesse condizioni iniziali, confrontare A con 0.7162450345895195 entro
   tolleranza numerica. Questo è f(1), non la baseline SHAP f(0).

Rischi residui: lo snapshot non valida da solo la fedeltà temporale del caso;
clonare loader/cache ha un costo di memoria e tempo; la riproducibilità numerica
GPU non è verificata. Eventuali feature originali discordanti per occorrenze
non neutralizzate dello stesso nodo restano soggette alla semantica del
GraphReindexer esterno. Nessun refactoring o modifica di external/orthrus,
mapping, sidecar, database o core Kernel SHAP è incluso.

### Implementazioni precedenti

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
