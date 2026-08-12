# Perturbazioni ORTHRUS

## Scopo del modulo

`OrthrusPerturbationManager` applica una maschera SHAP ai gruppi in `component_to_edges` e restituisce un nuovo `OrthrusAlertCase`. Offre due semantiche diverse: rimozione fisica degli edge oppure neutralizzazione delle feature.

## Posizione nella pipeline

Riceve il caso già arricchito dal builder e precede la chiamata al modello. Per ORTHRUS reale la strategia preferita è `neutralize_edges`; `mask_edge_features` è un alias normalizzato allo stesso comportamento.

## Come funziona

`drop_edges` raccoglie gli indici delle componenti attive, copia superficialmente il `TemporalData`, filtra i campi allineati agli edge (`src`, `dst`, `t`, `msg`, `edge_type`, `edge_feats`, `x_src`, `x_dst`) e ricostruisce `edge_index` da `src` e `dst`. Filtra anche i mapping leggibili e annota componenti attive e rimosse. `full_data` resta invariato.

Questa modalità è utile con dati sintetici e modelli non stateful perché produce un sottografo intuitivo. È però rischiosa nel percorso reale. In termini semplici, `LastNeighborLoader` mantiene memoria degli edge inseriti nel tempo e assegna loro un `e_id` globale; il modello può usare quell'identificativo per cercare le feature nell'intero `full_data`. Se un edge viene rimosso, il contatore o la sequenza possono slittare: un `e_id` può quindi puntare alla riga sbagliata di `full_data`. `drop_edges` non è sicuro senza supporto esplicito al global `e_id` o un replay temporale controllato.

`neutralize_edges` evita lo slittamento. Conserva invariati `src`, `dst`, `t`, `edge_index` ed `edge_type`, quindi mantiene numero, ordine e topologia degli edge. Copia e azzera invece le righe inattive di `msg`, `x_src`, `x_dst` ed `edge_feats`. Gestisce liste, array NumPy e tensori Torch, preservando per questi ultimi dtype e device.

## Perché è stato progettato così

La neutralizzazione cerca di rendere una componente meno informativa senza cambiare la storia temporale osservata dal loader. È un compromesso più compatibile con l'indicizzazione globale di ORTHRUS rispetto alla rimozione locale.

## Stato attuale

Entrambe le strategie sono implementate e testate sinteticamente. I test Torch della neutralizzazione verificano dtype, device e contenuto azzerato. Il gestore mantiene l'interfaccia `.dataset` richiesta dall'explainer corrente.

## Limiti e TODO

Neutralizzare non elimina l'edge: il modello vede ancora endpoints, tempo e tipo di relazione, quindi la componente conserva informazione strutturale. Non è ancora dimostrato che azzerare tutti e quattro i campi corrisponda alla baseline semantica corretta del modello reale. Anche alcuni commenti interni a `drop_edges` lo descrivono come sicuro per ORTHRUS, affermazione superata dalla cautela su `e_id/full_data`. Servono test con adapter fake stateful e poi replay reale controllato.

## File collegati

`perturbation/orthrus_perturbation_manager.py`, `preprocessing/orthrus_alert_case.py`, `perturbation/orthrus_interpretable_builder.py` e i moduli ORTHRUS futuri relativi a `LastNeighborLoader`.
