# Adapter del modello

## Scopo del modulo

Un adapter espone al livello XAI una sola operazione: trasformare un input perturbato in uno score di anomalia scalare. Nasconde caricamento, preparazione e dettagli del modello dietro `predict_anomaly_score`.

## Posizione nella pipeline

Kernel SHAP valuta molte maschere. Per ogni maschera, il gestore crea un input perturbato e l'adapter restituisce un numero reale. Senza questa riduzione scalare, `shap.KernelExplainer` non può spiegare la funzione black-box prevista dal progetto.

## Come funziona

`OrthrusAnoAdapter` definisce il contratto astratto. La factory `build_adapter` seleziona dummy, placeholder o real.

`DummyOrthrusAnoAdapter` accetta soltanto `LogDataset`. Conta record e path distinti, assegna pesi artificiali a parole chiave nei path e negli argomenti, aggiunge un piccolo termine deterministico dipendente dal seed e passa il risultato a una sigmoide. Il dataset vuoto vale zero. È intenzionalmente semplice e non riproduce ORTHRUS.

`PlaceholderOrthrusAnoAdapter` produce un errore esplicativo. `RealOrthrusAnoAdapter` riceve invece un modello già costruito tramite dependency injection. Verifica `OrthrusAlertCase`, `temporal_data` e `full_data`, sposta i due oggetti sul device quando espongono `.to(device)`, richiede la modalità `eval` quando configurato ed esegue la chiamata sotto `torch.no_grad()` quando Torch è disponibile.

Il contratto implementato chiama esattamente `model(batch, full_data, inference=True)`, valida un output numerico per-edge e calcola `mean(edge_losses)`, così ogni perturbazione produce un singolo score confrontabile da SHAP. Il numero di edge viene ricavato da `edge_index`, poi da `src` o `msg` come fallback.

## Perché è stato progettato così

Isolare l'adapter mantiene invariato Kernel SHAP quando cambia il modello e permette di testare prima il contratto con un fake. Impedisce inoltre che mapping descrittivi, checkpoint e loader temporali invadano il codice dell'explainer.

## Stato attuale

Dummy e placeholder sono operativi. Anche il core finale del real adapter è implementato e testato con modelli fake e, quando disponibile, tensori Torch. Non costruisce né carica autonomamente il modello.

## Limiti e TODO

Mancano costruzione del modello, caricamento del checkpoint e configurazione, loader/replay temporale, gestione reale di neighbor state ed `e_id` e smoke test con artifact. Il core assume che `case.temporal_data` sia già il batch accettato dal modello e che l'output diretto della chiamata sia il vettore delle loss, non una tupla o un dizionario.

## File collegati

`adapters/orthrus_ano_adapter.py`, `adapters/orthrus_runtime.py`, `preprocessing/orthrus_alert_case.py`, `pipeline.py`, `xai/kernel_shap_explainer.py` e `docs/ORTHRUS_INTEGRATION.md`.
