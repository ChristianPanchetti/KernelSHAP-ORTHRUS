# Mapping, join key e sidecar

## Scopo del modulo

Questi moduli traducono gli indici tecnici degli edge in riferimenti comprensibili, per esempio UUID dell'evento, operazione e metadata originali. Il mapping migliora la spiegazione, ma non alimenta l'inferenza del modello.

## Posizione nella pipeline

Il mapping può arricchire un `OrthrusAlertCase` dopo il caricamento e può essere consultato quando si descrivono le componenti più importanti. È separato da `full_data`, che appartiene invece al contesto numerico necessario a ORTHRUS.

## Come funziona

Un **edge index** è la posizione locale di un edge nel `TemporalData` o nel caso spiegato. Un **`e_id`** è l'identificativo globale/incrementale usato dal percorso temporale e dal neighbor loader per recuperare feature. Un **`event_uuid`** è l'identità dell'evento originale nel dominio DARPA/ORTHRUS. Non sono intercambiabili: in particolare `e_id` non è `event_uuid`.

`ArtifactOnlyMappingProvider` legge ciò che è già disponibile nell'artifact, come sorgente, destinazione, tempo e tipo, e popola metadata best effort. È sempre il fallback più semplice, ma non recupera UUID persi durante la featurizzazione.

`SidecarMappingProvider` carica un JSON indicizzato per edge locale, verifica struttura e copertura e trasferisce UUID e metadata nel caso. È deterministico soltanto se il sidecar è stato prodotto con lo stesso ordine degli edge del `TemporalData`.

`DbAssistedMappingProvider` può oggi confrontare in memoria edge e righe sintetiche. La parte live non interroga Postgres e solleva esplicitamente `NotImplementedError`. `GraphAssistedMappingProvider` è anch'esso un placeholder.

`orthrus_join_keys.py` normalizza sorgente, destinazione, timestamp e, quando decodificabile, operazione. L'operazione può provenire da un campo esplicito o da `edge_type`; se la codifica non è interpretabile il codice richiede istruzioni ulteriori. Join senza operazione possono essere ambigue.

`OrthrusSidecarGenerator` può generare un sidecar da un grafo pre-embedding, mantenendo l'ordine di iterazione degli edge, oppure unire un `TemporalData` a righe evento offline. Registra match, mancanti, collisioni, fallback e warning; offre anche una validazione rispetto al numero di edge.

## Perché è stato progettato così

ORTHRUS può perdere `event_uuid` quando converte il grafo in `TemporalData`. Conservare il collegamento in un file separato evita una dipendenza Postgres durante ogni spiegazione e non mescola dati per analisti con feature usate dal modello.

## Stato attuale

Artifact-only, sidecar, join offline, generazione e validazione sono implementati e testati con dati sintetici. Le descrizioni di componenti possono includere esempi di edge mappati.

## Limiti e TODO

Restano futuri Postgres reale, graph-assisted mapping operativo e validazione con artifact/grafi ORTHRUS reali. Collisioni su `(src, dst, t)` richiedono l'operazione o ulteriori chiavi. I sidecar generati sono ignorati da Git per default perché possono contenere dati sensibili e dipendono dagli artifact locali.

## File collegati

`preprocessing/orthrus_mapping.py`, `preprocessing/orthrus_join_keys.py`, `preprocessing/orthrus_sidecar_generator.py`, `preprocessing/orthrus_alert_case.py` e `docs/ORTHRUS_INTEGRATION.md`.
