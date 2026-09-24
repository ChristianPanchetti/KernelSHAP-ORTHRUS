# Diario implementativo pipeline ORTHRUS

## Obiettivo finale

La pipeline finale deve spiegare con Kernel SHAP uno score prodotto da ORTHRUS-ano su artifact nativi `TemporalData`. Il percorso previsto costruisce un `OrthrusAlertCase`, genera componenti interpretabili, neutralizza le componenti inattive preservando ordine ed edge identity, esegue ORTHRUS e associa i contributi SHAP agli edge e ai metadata leggibili.

## Stato prima di questo step

All'inizio del percorso implementativo il progetto disponeva della pipeline dummy completa, del contenitore ORTHRUS, del builder, delle perturbazioni e dei mapping sintetici. `RealOrthrusAnoAdapter` esponeva soltanto metodi stub: non poteva invocare un modello né ridurre gli edge loss. Anche la modalità ORTHRUS di `pipeline.py` era non implementata. Le fasi documentate sotto descrivono il successivo completamento.

## Step corrente

### Visualizzazioni delle attribuzioni Kernel SHAP (2026-09-24)

Aggiunto `xai/result_visualizer.py`: rendering matplotlib indipendente da SHAP,
ORTHRUS e PostgreSQL. Accetta `KernelSHAPResult` tramite `to_dict()` oppure il
JSON esistente, senza modificarne il formato e senza ricalcolare attribuzioni.
Utilizza `components[].component_id/shap_value`, `baseline_score`,
`original_anomaly_score` e `metadata.case.component_mapping/node_mapping`.
Non considera i nomi liberi delle componenti una prova di identità verificata.

Il bar plot orizzontale mostra tutte le componenti in ordine stabile decrescente
per valore assoluto, mantenendo segno e valore numerico. Zero è visibile; rosso
indica contributi positivi, blu negativi, grigio nulli. OTHER rimane una sola
barra. Nessun top-k grafico o raggruppamento ulteriore. L'altezza cresce con il
numero di componenti e le righe delle etichette.

Il waterfall usa lo stesso ordine e le stesse etichette: parte da f(0), mostra
ogni incremento firmato e termina con barre distinte per f(0)+sum(phi) e f(1).
Il titolo riporta sempre il residuo f(1)-[f(0)+sum(phi)], anche quando nullo.
Non introduce una componente per il residuo e non modifica i contributi per
far coincidere i due score. Baseline e score ricostruito possono essere negativi.
La somma è ricalcolata soltanto dai numeri esportati per disegnare il percorso;
non viene rieseguito Kernel SHAP. Valori non finiti, componenti vuote o ID
ripetuti causano un errore esplicito.

Le etichette includono sempre il component_id. Path/cmd sono aggiunti soltanto
per una componente con una sola source, nodo `resolved`, provenienza Fase 10
`postgresql` o `offline_rows`, campo non dichiarato `unverified_fields`.
Negli altri casi viene mostrato il solo ID. Non viene inferita un'identità da
name/description, UUID o eventi incidenti. I dettagli lunghi sono abbreviati
con ellissi e disposti su più righe; il JSON originale conserva il testo completo.
OTHER è etichettata come aggregata. Il rendering non certifica autonomamente
il contenuto delle righe offline: conserva il contratto e gli stati della Fase 10.

Al termine del ramo ORTHRUS della pipeline, dopo mapping e salvataggio JSON,
vengono generati automaticamente quattro file affiancati al JSON:
`<stem>_shap_bar.png`, `<stem>_shap_bar.pdf`,
`<stem>_shap_waterfall.png`, `<stem>_shap_waterfall.pdf`.
Il ramo dummy rimane invariato. Matplotlib è dichiarato in requirements.txt;
nessuna finestra interattiva è richiesta (FigureCanvasAgg). PNG a 160 dpi, PDF
vettoriale, layout adattivo e bounding box completo. La generazione separata
sovrascrive soltanto i quattro grafici omonimi, non il JSON.

Dalla CLI esistente, dopo aver installato le dipendenze:

```bash
python main.py --plot-json outputs/theia_e5_phase11.json
```

Questo ramo termina prima di importare/eseguire la pipeline; le opzioni di
inferenza non vengono utilizzate. Da Python: `export_plots(result, json_path)`
o `export_plots_from_json(json_path)` nel modulo di visualizzazione.

**Test locali mirati:** `tests/test_result_visualizer.py` controlla dati sintetici,
ordine, contributi firmati, baseline non nulla, fallback delle etichette,
provenienza/campi non verificati, OTHER, numerose componenti, input non validi
e instradamento CLI senza pipeline. Include inoltre test del rendering,
coordinate waterfall/residuo, file PNG/PDF, JSON invariato e import bloccati
per modello/SHAP/database. Questi ultimi richiedono matplotlib.

Comando eseguito: `python -m pytest -q tests/test_result_visualizer.py --basetemp=.pytest_plots_final_20260924 --tb=short`.
Esito: **13 passed, 5 skipped** (tutti gli skip richiedono matplotlib).

In questo ambiente matplotlib non è installato e i tentativi di download
non sono riusciti (restrizioni socket/connessione e indice senza distribuzioni
accessibili). Pertanto i test di rendering vengono esplicitamente saltati;
la verifica visiva e l'effettiva produzione PNG/PDF restano da eseguire in un
ambiente con la dipendenza disponibile. Non è stata simulata una verifica grafica.
Nessun accesso al server, al modello reale o al database.

Bar plot e waterfall visualizzano attribuzioni numeriche: **non sostituiscono**
il sottografo attribuito o la heatmap temporale delle specifiche della tesi.
Restano pendenti la validazione PostgreSQL Fase 10, quella end-to-end Fase 11
e il controllo delle etichette/leggibilità sui risultati THEIA_E5 reali.

### Fase 11 — Kernel SHAP–ORTHRUS end-to-end (2026-09-24)

Completato il ramo ORTHRUS dell'orchestrazione esistente in `pipeline.py`.
Il percorso usa il KernelSHAPExplainer e ResultExporter già presenti:
preparazione ufficiale -> builder node-based -> score_mask isolato -> SHAP ->
un solo arricchimento DB-assisted -> descrizioni e JSON. Nessun nuovo motore
SHAP, adapter, modello, dashboard o formato di export.

**Raccordo additivo alla Fase 9.** `prepare_official_orthrus_case` restituisce
`(case, model, rel2id, warnings)` tramite il percorso ufficiale già esistente.
L'opzione additiva `prepare_only=True`, sempre con `perturbative=True`, restituisce
gli oggetti subito dopo `_prepare_temporal_batch`, prima di A1/B/A2/Z. Non
modifica il caricamento di cfg/artifact/checkpoint, il controllo della storia,
il prefisso validation/test, il contatore o le verifiche di allineamento.
La modalità `--phase9` continua a eseguire le quattro valutazioni come prima.
Il ramo SHAP non ripete quelle quattro valutazioni. Il ripristino della directory
corrente del runtime resta nel medesimo `finally`.

La pipeline crea un unico manager `neutralize_edges` e un unico
`RealOrthrusAnoAdapter(..., device=None, isolate_state=True)` sul modello e sul
batch già predisposti al device dalla preparazione ufficiale. Tutti i forward
del batch selezionato passano da `adapter.score_mask(manager, mask)`, serialmente,
con lo stesso snapshot pre-batch. Nessuna modifica ad adapter, neutralizzazione,
LastNeighborLoader, GraphReindexer, snapshot/restore, edge losses, full_data,
ordine, topologia o target. Nessuna modifica a `external/orthrus`.

**Componenti e maschere.** Il builder esistente costruisce componenti source-based.
La pipeline le converte in InterpretableComponent/InterpretableSpace nell'ordine
canonico del dizionario: mask[j], component_id[j] e shap_values[j] rimangono
allineati. Il ranking riordina solo la presentazione. Controlli prima dello scoring
SHAP rifiutano componenti vuote, indici invalidi, sovrapposizioni nell'assegnazione
degli edge e copertura incompleta. Il limite 1 è consentito se esiste una sola
componente; se il builder produce nodo + OTHER viene rifiutato esplicitamente.
Il builder e la politica di perturbazione non sono stati modificati.

OTHER mantiene un singolo valore SHAP per l'aggregato. Gli interventi sulle
feature possono sovrapporsi per i nodi ripetuti nello stesso ruolo: non sono
interventi indipendenti sugli eventi originali e non sono prove causali.
Con la partizione completa del builder, tutti zero neutralizza tutte le righe
x_src/x_dst, mantenendo storia, topologia, messaggi e target. I metadata della
Fase 10 descrivono le righe coinvolte dalla disattivazione di una componente,
non una singola perturbazione rappresentativa di tutte le coalizioni SHAP.

**Scorer e diagnostica.** Il raccordo opzionale `mask_scorer` dell'explainer
è usato sia per gli estremi sia per le coalizioni della libreria. La cache è
condivisa: f(0) e f(1) non vengono rivalutati quando richiesti da SHAP. Il percorso
legacy resta disponibile se lo scorer opzionale non viene passato.
`shap.KernelExplainer` mantiene background zero, input uno e lo stesso kernel,
sampling e solver della libreria: nessuna normalizzazione delle attribuzioni.

f(0) è la baseline SHAP corrente, f(1) lo score originale corrente. Non viene
usato alcuno score storico della Fase 9 come costante. Si esportano expected_value,
somma dei contributi, residuo f(1)-f(0)-sum(phi), seed, budget richiesto,
`evaluated_unique_masks` (forward effettivi richiesti allo scorer),
`effective_nsamples` (nsamplesAdded della libreria, quando disponibile), versione
SHAP e l1_reg effettivo. Non sono conteggiati nei forward SHAP quelli del prefisso
temporale. expected_value corrisponde al background unico; f(0) può essere non
nullo e maggiore di f(1).

Score e valori SHAP devono essere finiti; la cardinalità delle attribuzioni deve
coincidere con il numero di componenti e expected_value deve essere scalare.
Un residuo oltre `1e-8 + 1e-6 * max(abs(f1), abs(f0), abs(sum(phi)))` produce un
warning esplicito, salvato nel JSON. Viene segnalato anche un expected_value
discordante da f(0). I valori non vengono corretti e lo scarto non è attribuito
automaticamente al sampling. Il default l1_reg resta quello della libreria,
registrato nell'output; la versione sul server deve essere controllata.

**Mapping Fase 10.** Dopo SHAP il provider esistente arricchisce una volta
l'unione di tutte le componenti; top_k limita solo il riepilogo. Non ci sono
query nello scorer. I metadati sono associati per component_id, con node_mapping,
component_mapping, edge_to_original_metadata, edge_to_event_uuid e mapping_quality.
Restano visibili missing, ambiguous, unresolvable e conflict. Il provider non
riutilizza UUID non verificati. I campi di rete restano disabilitati per default,
come nella Fase 10; questa CLI minima non li certifica né li abilita.

PostgreSQL usa `db_config={}` e le impostazioni libpq predisposte fuori dal
repository (ambiente/service file). Non si aggiungono credenziali ai JSON o ai
log. Un errore di mapping richiesto si propaga e non viene esportato un nuovo
risultato dichiarato completo. Per verifiche offline, `--mapping-rows` legge
un JSON con array `events` e `nodes` nel contratto Fase 10; devono contenere
tutti i candidati delle chiavi richieste. Non sono state modificate le funzioni
di mapping. La pipeline non genera nuovi sidecar o artifact.

**Risultato.** KernelSHAPResult e ResultExporter rimangono i contenitori/exporter
esistenti. Il JSON include ordine canonico, valore e segno, ranking, descrizione
arricchita, score, diagnostica, metadati originali, qualità/provenienza, device,
raggruppamento e coordinate temporali del caso. `record_indices`/`num_records`
sono riutilizzati come posizioni/conteggio degli edge locali del batch; la
semantica e `component_edges` sono dichiarati nei metadata. Non sono UUID, indici
nel grafo o e_id. Le sintesi IT/EN includono path/cmd disponibili nei nomi leggibili
e precisano la semantica degli interventi; OTHER rimane aggregata. Gli oggetti
frozen vengono aggiornati tramite dataclasses.replace dopo il mapping.

**Avvio dalla CLI esistente**, dalla root KSHAP_ORTHRUS sul server Linux:

```bash
python main.py --mode orthrus --adapter real \
  --orthrus-config examples/orthrus_theia_e5_real_smoke.json \
  --grouping-mode node --perturbation-mode neutralize_edges \
  --max-components 8 --num-samples 256 --seed 0 \
  --mapping-backend postgresql --output outputs/theia_e5_phase11.json
```

Il path JSON è quello del caso reale già predisposto sul server, non creato qui.
Deve indicare THEIA_E5/test/0/0, checkpoint e device effettivi. I path relativi
seguono la convenzione dello smoke esistente (directory di lancio). Predisporre
prima la connessione libpq al database corretto, senza credenziali versionate.
La CLI ORTHRUS usa per default node, neutralize_edges, max_components=8 e adapter
real; il legacy mantiene exec_path, drop_records, limite 50 e adapter dummy.
Le opzioni legacy incompatibili vengono rifiutate; `--input` non carica artifact
nel percorso ORTHRUS, che usa esclusivamente `--orthrus-config`. Il device viene
dal JSON ufficiale. Il budget 256 copre le 254 coalizioni interne quando M=8,
ma il costo reale non è stato misurato.

Per usare righe offline sostituire soltanto `--mapping-backend postgresql` con
`--mapping-backend offline --mapping-rows /percorso/righe_verificate.json`.
Il modello rimane reale: offline riguarda il recupero dei metadati, non una
simulazione dell'inferenza. I test locali iniettano esplicitamente un modello
simulato e non sono presentati come esecuzioni THEIA_E5.

**Verifica locale.** Test mirati eseguiti con SHAP reale e modello stateful
simulato, adapter e manager esistenti, mapping offline e controlli di esportazione.
Lo scorer sintetico ha baseline 5, contributi [4, -1] e interazione tra componenti
tramite destination ripetuta: verifica ordine non alfabetico, isolamento,
riuso della cache, un solo mapping dopo i forward, stati missing/ambiguous e JSON.
Sono coperti anche OTHER, componente singola, componenti vuote, limite 1,
fallimento del mapping obbligatorio, output non finiti/cardinalità errata e
residuo segnalato. Il nuovo test runtime verifica il ritorno prima delle quattro
valutazioni; sono stati riutilizzati i test sintetici del runtime/CLI e le
regressioni legacy di export, additività e riproducibilità.

```text
python -m pytest -q tests/test_orthrus_phase11.py tests/test_orthrus_official_runtime.py tests/test_orthrus_official_smoke_script.py tests/test_pipeline_smoke.py tests/test_shap_additivity.py tests/test_reproducibility.py --basetemp=.pytest_phase11_20260924_b --tb=short
```

Esito: **33 passed**, nessuno skip. Il primo tentativo senza basetemp aveva
incontrato errori di permesso nella directory temporanea Windows; usando una
directory del workspace i test sono stati eseguiti. Nessuna suite completa,
nessun accesso PostgreSQL, nessuna esecuzione ORTHRUS reale o rigenerazione artifact.

**Validazione reale ancora pendente:** prima validare il mapping PostgreSQL
Fase 10 sullo schema e sugli artifact del server; poi eseguire questa pipeline
su THEIA_E5 test/0/0 con lo stesso percorso temporale. Controllare finitezza,
expected_value, residuo, costo delle coalizioni e coerenza component_id -> SHAP ->
metadati/ruoli. Non serve rieseguire sistematicamente A1/B/A2/Z; la preparazione
temporale validata viene comunque eseguita. Torch/CUDA e il modello reale non
sono stati esercitati in Windows. Nessun commit o push.

### Fase 10 — DB-assisted mapping (2026-09-24)

Implementato l'arricchimento delle spiegazioni nel provider esistente, dopo
la verifica mirata dei file indicati dall'audit. Nessuna modifica a ORTHRUS,
TemporalData/full_data, runtime, adapter, perturbazione, snapshot/restore o
core Kernel SHAP. Nessun collegamento SHAP end-to-end e nessuna rigenerazione
degli artifact. La validazione reale della Fase 9 del 21 settembre resta valida
nel perimetro documentato sotto.

**Nodi.** `DbAssistedMappingProvider.enrich_case` raccoglie gli `index_id`
distinti negli estremi degli edge delle componenti richieste, oppure di tutto
il caso se `component_ids` è omesso. Interroga `subject_node_table`,
`file_node_table`, `netflow_node_table`; il tipo è associato alla tabella,
non è una nuova colonna SQL. Conserva in `case.metadata["node_mapping"]`
UUID, tipo, path, cmd e stato `resolved`, `missing`, `ambiguous` o
`unresolvable`. Più righe, anche nella stessa tabella, non producono un UUID
arbitrario. L'identità del nodo può essere risolta anche con path/cmd mancanti,
elencati in `missing_fields`.

Le colonne di rete `src_addr`, `src_port`, `dst_addr`, `dst_port` sono lette
ma non pubblicate per default. Il parametro `verified_network_fields`
consente esclusivamente le colonne già controllate dal chiamante sul dataset
reale; le altre sono indicate come `unverified_fields`. Il controllo è una
precondizione esterna, non una certificazione automatica dei valori. Questo
limite è necessario perché il parser THEIA_E5 locale assegna erroneamente
caratteri dell'UUID agli endpoint remoti in un ramo di `store_netflow`.

**Eventi.** Le posizioni di `component_to_edges` sono indici locali del caso.
Le chiavi di join sono `(src_index_id, dst_index_id, timestamp_rec, operation)`;
il timestamp THEIA_E5 resta un intero in nanosecondi. L'orientamento è già quello
del DB ORTHRUS, incluse le inversioni eseguite dal parser: il provider non
inverte nuovamente gli estremi. Il numero del nodo non è un UUID. `_id` SQL,
`event_uuid`, indice nel grafo e `global_edge_offset` restano distinti;
gli `e_id` del neighbor loader non vengono usati.

`orthrus_join_keys.py` decodifica one-hot Python/NumPy/Torch con il `rel2id`
ufficiale fornito esplicitamente. Il vettore usa la posizione `rel2id[label]-1`;
i valori numerici scalari, se forniti, sono invece ID della tabella `rel2id`,
non indici argmax a base zero. Non è incorporato un vocabolario EVENT_*.
Senza vocabolario, con vettori malformati o chiavi incomplete l'evento è
`unresolvable`. Per un artifact con sola `msg` serve anche `edge_type_slice`
ricavata dalla configurazione/layout effettivi, non stimata.

Il provider assegna `edge_to_event_uuid` soltanto per identità risolte.
Più righe compatibili danno `ambiguous`, zero righe `missing`; i dettagli
restano in `edge_to_original_metadata`, le statistiche separate per nodi ed
eventi in `mapping_quality`. `fail_on_unmatched=True` richiede la risoluzione
di tutti gli eventi selezionati, non la completezza dei metadati dei nodi.
Gli eventi compressi dal preprocessing non vengono ricostruiti: si cerca
soltanto l'evento rappresentante dell'edge.

**PostgreSQL e offline.** `rows` contiene gli eventi; `node_rows` contiene i
nodi con `index_id`, `node_uuid`, `node_type` (`subject`, `file`, `netflow`) e
le colonne pertinenti. Le righe fornite devono includere TUTTI i candidati
per le chiavi richieste: un export troncato non dimostra unicità. Gli iterable
sono materializzati una volta. È possibile fornire soltanto i nodi offline.

Con `db_config` il driver opzionale `psycopg2` viene importato solo al fetch.
Il provider apre e chiude una connessione propria in transazione read-only,
usa colonne esplicite, parametri SQL, ID/chiavi distinti e blocchi da 200.
Le ricerche nodo usano `index_id = ANY(%s)`; gli eventi usano le tuple complete
o gli UUID già verificati. Gli indici evento VARCHAR ricevono parametri stringa,
senza cast della colonna. Il timeout delle istruzioni è 30 secondi. Non vengono
creati indici o modificato lo schema. Gli errori pubblici non includono DSN,
host o messaggi del driver potenzialmente contenenti credenziali.

Nessuna connessione PostgreSQL reale è stata tentata in Windows. I vincoli
WHERE non garantiscono un accesso indicizzato: il DDL locale non dichiara
indici adatti su index_id/event_uuid/chiave evento. Prima del controllo reale
servono schema effettivo e EXPLAIN. Non eseguire scansioni massive per
aggirare indici mancanti. La corrispondenza tra importazione DB e artifact è
una precondizione da verificare sul server; gli index_id possono cambiare con
una nuova importazione.

Esempio offline, dopo aver costruito `case.component_to_edges`:

```python
provider = DbAssistedMappingProvider(rel2id=orthrus_rel2id)
provider.enrich_case(case, rows=event_rows, node_rows=node_rows,
                     component_ids=selected_component_ids)
text = describe_component(case, selected_component_ids[0])
```

Per PostgreSQL sostituire con `DbAssistedMappingProvider(db_config={},
rel2id=orthrus_rel2id)` e omettere `rows/node_rows`: `{}` usa la configurazione
libpq predisposta fuori dal sorgente, per esempio ambiente/service file.
Non stampare la configurazione di connessione. Passare un iterable vuoto
esclude il fetch di quel tipo di righe. `orthrus_rel2id` deve provenire dal
checkout effettivamente usato per gli artifact.

L'arricchimento va chiamato una volta prima o dopo SHAP, per l'unione delle
componenti richieste. Non è collegato alla funzione di score. I dizionari del
caso sono riutilizzabili dalle descrizioni senza query; chiamate successive
sostituiscono il precedente scope di arricchimento. Non è stata introdotta
una cache persistente condivisa tra database diversi.

**Sidecar e provenienza.** Il generatore esistente accetta
`generate_from_graph(graph_path, output_path, temporal_data_path=...)`:
confronta l'intera sequenza delle tuple del grafo con l'artifact e registra
SHA-256 del file TemporalData e scope `graph`. Il caricamento verificato usa:

```python
SidecarMappingProvider(
    sidecar_path,
    temporal_data_path=graph_temporal_data_path,
    graph_edge_offset=batch_start_in_graph,
    rel2id=orthrus_rel2id,
    edge_type_slice=original_msg_edge_type_slice,
).enrich_case(case)
```

`graph_edge_offset` è obbligatorio nel percorso verificato e va ricavato dalla
selezione reale del batch, non da `global_edge_offset`. Il loader verifica
hash del file, cardinalità, corrispondenza della slice con il batch e tuple
per edge. Chiavi duplicate nell'artifact restano prudentemente ambigue.
Per artifact già dotati di `edge_type` lo slice non serve. L'hash verifica la
coerenza dell'artifact, non autentica un sidecar proveniente da fonti non fidate.

La generazione da righe assegna UUID solo su candidato unico; una discordanza
di operazione non viene più ignorata. La provenienza verificabile viene
registrata soltanto per join completi con operazione disponibile. Il vecchio
fallback senza operazione resta un percorso legacy non verificato. Anche i
sidecar legacy caricati senza `temporal_data_path` restano leggibili, ma sono
marcati `unverified` e il provider DB non riutilizza i loro UUID come identità
provate. La vecchia `validate_sidecar` controlla la struttura/copertura, non
sostituisce il caricamento verificato.

Solo gli UUID provenienti da sidecar verificati vengono riutilizzati direttamente;
un precedente join DB viene ricontrollato sulle nuove righe per non nascondere
eventuali nuovi candidati ambigui. Gli UUID verificati sono legati alle tuple ordinate e alle coordinate disponibili
del caso tramite fingerprint. Un caso cambiato non riutilizza quelle identità.
Il DB può poi cercare direttamente l'UUID e controllare comunque gli estremi:
righe duplicate restano ambigue, tuple discordanti danno `conflict`. Un UUID
già provato dal grafo può restare risolto anche senza riga DB, ma
`database_status` distingue `missing` e `not_queried` dall'effettivo recupero.

**Descrizioni node-based.** `component_mapping` conserva edge assegnati,
nodi source/destination e righe `x_src/x_dst` coinvolte se la componente viene
disattivata. La descrizione legge UUID, tipo, path/cmd, endpoint autorizzati,
operazione, timestamp, stato e provenienza. `OTHER` può contenere più nodi.
Il raggruppamento per source non viene esteso a tutti gli edge incidenti.
Le righe neutralizzabili seguono `all_occurrences_per_node_role`, senza
propagazione ricorsiva o tra ruoli; possono appartenere a componenti attive.
Quando disponibili, i metadata Fase 9 della mask corrente sono mostrati
separatamente come righe effettivamente neutralizzate. Nessun evento è eliminato.

**Verifica locale mirata.** Eseguiti i test dei tre moduli mapping/sidecar già
esistenti e il nuovo `tests/test_orthrus_db_mapping.py`, senza suite completa,
checkpoint o THEIA_E5 reale. Copertura: identità nodi e campi mancanti,
collisioni, one-hot, orientamento, eventi incompleti, politica dei nodi ripetuti,
OTHER, sidecar fuori batch/artifact, grafo riordinato, UUID duplicati, query
parametrizzate selettive/read-only e assenza di credenziali negli errori.

```text
python -m pytest -q tests/test_orthrus_db_mapping.py tests/test_orthrus_mapping.py tests/test_orthrus_mapping_sidecar.py tests/test_orthrus_sidecar_generator.py
```

Esito: **32 passed, 1 skipped**. Lo skip è il controllo Torch, non installato
nel Python locale; il percorso NumPy one-hot è verificato. Il driver è simulato
nei test SQL: compatibilità PostgreSQL, piani di esecuzione, dati reali e
caricamento Torch degli artifact restano da validare sul server.

**Controllo reale minimo successivo:** verificare schema/indici e provenienza
DB-artifact; usare il caso test/0/0 da 1024 edge senza nuova inferenza; selezionare
una componente source e pochi suoi edge, passare il rel2id effettivo; arricchire
una volta; confrontare UUID/path/cmd e candidati evento con le righe DB,
controllando orientamento e nanosecondi. Se disponibile, verificare il sidecar
contro il grafo/artifact corrispondente e offset nel grafo. Controllare che
ambiguità e valori mancanti siano riportati e che tensori, full_data e score
non siano coinvolti. Abilitare i campi di rete solo dopo un controllo dei dati
importati. Nessun test reale, commit o push in questo step.

### Fase 9 completata — validazione reale THEIA_E5 (2026-09-21)

La Fase 9 è stata validata sul caso THEIA_E5 selezionato il **21 settembre
2026**, nel container Linux ORTHRUS con ambiente Python **pids**. I risultati
riportati qui provengono dall'esecuzione sul server comunicata dall'utente;
non è stata rieseguita alcuna inferenza nella repository Windows.
L'implementazione della perturbazione e dello snapshot/restore del 17 settembre
e la preparazione temporale del 20 settembre sono distinte da questa
validazione reale. Le sezioni datate precedenti conservano lo stato storico
dei lavori, superato per questo caso dall'esito positivo riportato qui.

Configurazione e stato immediatamente precedente al batch:

| Parametro | Valore |
| --- | --- |
| dataset | THEIA_E5 |
| split | test |
| graph_index | 0 |
| batch_index | 0 |
| device | cuda |
| numero di edge | 1024 |
| global_edge_offset | 16351122 |
| train_edges | 12385364 |
| prefix_batches | 3921 |

Il runtime ha verificato la memoria del checkpoint rispetto agli edge train,
attraversato il prefisso temporale di validation e acquisito lo snapshot
immediatamente prima di test/0/0. Le quattro valutazioni sono partite dallo
stesso snapshot iniziale ORTHRUS.

| Valutazione | Mask | Score | Edge losses |
| --- | --- | --- | --- |
| A1 | Tutte le componenti attive | 0.7171946001362812 | 1024 |
| B | Componente node:1301260 inattiva | 0.7381046357841115 | 1024 |
| A2 | Tutte le componenti attive nuovamente | 0.7171945968802902 | 1024 |
| Z | Tutte le componenti selezionate inattive | 0.8296128124493407 | 1024 |

Gli score sono finiti. La ripetibilità A1 ≈ A2 è stata verificata con
`rtol = 1e-6` e `atol = 1e-8`. I controlli di invarianza e di ripristino
dello stato ORTHRUS dopo ogni valutazione sono risultati soddisfatti.
La perturbazione modifica esclusivamente le feature `x_src` e `x_dst`:
`full_data`, topologia, numero e ordine degli edge e target `edge_type`
rimangono invariati. Non è richiesto che B o Z abbiano score inferiore ad A1.

La vecchia baseline **0.7162450345895195** non è direttamente confrontabile:
il precedente smoke test non documentava lo stesso stato temporale iniziale.
La nuova baseline di riferimento per **test/0/0 con percorso temporale
preparato è 0.7171946001362812** (A1).

Durante l'esecuzione reale era emerso l'errore
`'GlobalStorage' object has no attribute 'msg'`: la copia ORTHRUS sul server
usa una patch per ridurre il consumo di memoria e non costruisce
`full_data.msg` quando THEIA_E5 usa `edge_features = edge_type`.
La correzione, già applicata e validata sul server, è stata riportata
esattamente in `adapters/orthrus_runtime.py`, in `_prepare_temporal_batch`:
il ciclo di controllo passa da `("t", "edge_type", "msg")` a
`("t", "edge_type")`. Restano i controlli su `t`, `edge_type` e `cur_e_id`.
Non viene ricostruito `full_data.msg` e non viene modificato external/orthrus.

**Esito: Fase 9 completata e validata per il caso THEIA_E5 selezionato.**
Gli score ottenuti sono reali, ma **non sono ancora valori SHAP**.
Rimane da implementare il collegamento Kernel SHAP end-to-end.
Questo aggiornamento Windows riguarda soltanto il runtime e il diario;
nessun nuovo test, esperimento, commit o push è stato eseguito.

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

Al 21 settembre 2026 la Fase 9 è validata sul caso reale THEIA_E5 test/0/0
descritto sopra. Rimane da implementare il collegamento Kernel SHAP end-to-end
e la relativa orchestrazione in `pipeline.py`: gli score perturbativi validati
non sono ancora valori SHAP.

## Prossimi step previsti

- collegare builder, neutralizzazione, adapter isolato ed explainer Kernel SHAP
  nella modalità ORTHRUS, partendo dal percorso temporale reale validato.

## Cronologia aggiornamenti

- 2026-08-11: implementato il core di `RealOrthrusAnoAdapter` con model injection, chiamata di inferenza, validazione delle loss e riduzione media; aggiunti test fake e Torch opzionale.
- 2026-08-12: aggiunto il runtime ORTHRUS con import dinamico, controlli di disponibilità, caricamento configurabile e smoke test non perturbato; `pipeline.py` resta volutamente stub per Kernel SHAP ORTHRUS.
- 2026-08-12: l'audit del checkout ufficiale ha confermato la firma dell'adapter e l'output per-edge. Il runtime è stato esteso con una via ufficiale basata su `cfg`, `load_all_datasets`, `build_model`, `load_model` e `batch_loader_factory`. `full_data` viene ricostruito dai tre split; il checkpoint `model_epoch_N` ripristina sia `state_dict.pkl` sia `neighbor_loader.pkl`. Lo step copre una sola inferenza non perturbata: restano il test con artifact reali, la verifica temporale/device e Kernel SHAP end-to-end.
- 2026-08-12: aggiunti il template `examples/orthrus_official_smoke_config.example.json` e la CLI `scripts/run_orthrus_official_smoke.py`. Rendono ripetibile il primo smoke test reale senza includere dataset o checkpoint nel repository. La CLI esegue ancora soltanto un batch non perturbato; con gli artifact disponibili restano da validare caricamento, stato temporale, `e_id/full_data` e device prima dell'integrazione Kernel SHAP.
