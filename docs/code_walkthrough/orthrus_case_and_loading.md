# OrthrusAlertCase e caricamento

## Scopo del modulo

`OrthrusAlertCase` è il contenitore centrale di una spiegazione ORTHRUS: rappresenta una finestra, un batch o un alert locale senza obbligare il resto del framework a dipendere direttamente dalle classi Torch Geometric.

## Posizione nella pipeline

Si colloca dopo il caricamento degli artifact e prima di builder, perturbazioni e adapter reale. La pipeline finale usa `TemporalData` dentro questo oggetto, non `LogDataset`.

## Come funziona

`temporal_data` contiene il caso locale con edge cronologici e campi tipici `src`, `dst`, `t`, `msg` e, quando disponibili, `edge_type`, `edge_index`, `x_src`, `x_dst` ed `edge_feats`. `full_data` è invece il contesto globale usato dal modello per recuperare feature coerenti con gli identificativi degli edge. Non è metadata descrittivo.

`metadata` raccoglie informazioni operative sul caso e sulle perturbazioni. `component_to_edges` collega componenti SHAP a indici edge locali; gli altri dizionari collegano edge, record, nodi, UUID e metadata originali. `event_uuid` appartiene al livello di leggibilità e tracciabilità, non sostituisce `e_id`.

La proprietà `num_edges` prova prima la lunghezza di `src`, poi la seconda dimensione di `edge_index`. I tipi sono volutamente generici per mantenere importabile il progetto senza Torch/PyG.

`OrthrusAlertCaseLoader` verifica il path, seleziona un mapping provider, carica preferibilmente con `torch.load` su CPU e usa pickle come fallback sintetico. Costruisce poi il caso e applica il provider. Attualmente carica un solo artifact come `temporal_data`: non riceve né associa autonomamente `full_data`.

## Perché è stato progettato così

Il contenitore disaccoppia XAI e ORTHRUS. Questa separazione permette test sintetici piccoli e rende esplicito che il contesto del modello (`full_data`) è diverso dal contesto umano della spiegazione (mapping ed `event_uuid`).

## Stato attuale

Dataclass, conteggio edge, caricamento pickle e mapping artifact-only sono testati. Il caricamento Torch esiste in forma best effort ma non è stato validato con artifact ORTHRUS reali.

## Limiti e TODO

Mancano contratto e validazione dei campi richiesti dal modello, caricamento coordinato di `full_data`, configurazione/versione del checkpoint, device policy e smoke test reale. Il fallback pickle non rende automaticamente compatibili oggetti PyG complessi.

## File collegati

`preprocessing/orthrus_alert_case.py`, `preprocessing/orthrus_alert_case_loader.py`, `preprocessing/orthrus_mapping.py`, `perturbation/orthrus_interpretable_builder.py` e `adapters/orthrus_ano_adapter.py`.
