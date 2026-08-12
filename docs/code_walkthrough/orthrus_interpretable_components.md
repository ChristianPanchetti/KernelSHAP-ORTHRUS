# Componenti interpretabili ORTHRUS

## Scopo del modulo

`OrthrusInterpretableBuilder` trasforma gli edge di un `TemporalData` in gruppi binari comprensibili da Kernel SHAP. Non modifica il grafo: popola `case.component_to_edges`.

## Posizione nella pipeline

Opera dopo la costruzione di `OrthrusAlertCase` e prima del gestore delle perturbazioni. Ogni posizione della maschera SHAP corrisponde, secondo un ordine deterministico, a una chiave di `component_to_edges`.

## Come funziona

In modalità `edge`, ogni edge è una componente. In modalità `node`, il codice corrente crea una partizione usando il solo nodo `src`; non raggruppa contemporaneamente per entrambi gli estremi. In modalità `edge_type`, valori scalari sono usati direttamente e vettori simili a one-hot vengono ridotti con argmax. In modalità `time_chunk`, gli edge sono raggruppati rispetto al timestamp minimo.

Il nome `time_chunk_seconds` è più forte del comportamento effettivo: il valore viene usato nella stessa unità numerica di `temporal_data.t`, senza conversione automatica da secondi a nanosecondi. Se `max_components` limita i gruppi, quelli più grandi vengono mantenuti e il resto confluisce in `OTHER`. Il builder controlla infine che ogni edge compaia esattamente una volta.

## Perché è stato progettato così

Kernel SHAP lavora su vettori di presenza/assenza, mentre ORTHRUS lavora su edge temporali. `component_to_edges` è il ponte stabile fra i due domini e consente di cambiare la granularità interpretativa senza cambiare il modello.

## Stato attuale

Le quattro modalità sono implementate e verificate indirettamente dai test ORTHRUS di mapping e perturbazione. Il docstring sorgente che definisce ancora il builder “stub” è obsoleto rispetto al comportamento reale.

## Limiti e TODO

I test usano oggetti `FakeTemporalData`; non verificano tipi, unità temporali o codifica degli edge type di artifact reali. Va deciso se il grouping per nodo debba includere `src` e `dst`, e va formalizzata l'unità di `t`. Manca inoltre l'adattamento di queste componenti al tipo `InterpretableSpace` atteso dall'explainer legacy.

## File collegati

`perturbation/orthrus_interpretable_builder.py`, `preprocessing/orthrus_alert_case.py`, `preprocessing/orthrus_mapping.py` e `perturbation/orthrus_perturbation_manager.py`.
