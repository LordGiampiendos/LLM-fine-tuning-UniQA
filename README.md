# LLM-fine-tuning-UnipaQA

Questo repository contiene il codice per il fine-tuning di modelli di linguaggio utilizzando il dataset UniQA. Il dataset UniQA è disponibile su GitHub al seguente link: https://github.com/CHILab1/UniQA/tree/main.

## Struttura del Repository

Il repository è organizzato nel seguente modo:

- `Fine_Tuning/`: Contiene gli script e il codice necessari per il fine-tuning dei modelli.
- `Inferenza_In-context_Learning/`: Include gli script per l'inferenza utilizzando l'in-contenxt learning e i risultati ottenuti in termini di risposte generate e metriche calcolate.
- `Metriche-LLM-10-GPT/`: Contiene gli script per la valutazione con metriche di vautazione basate su IA (RAGAS/gpt-4o-mini).

## Requisiti

Assicurati di avere installato i seguenti pacchetti Python:

- jupyter
- transformers
- trl
- peft
- ragas

Puoi installare i requisiti necessari utilizzando `pip`:

```bash
pip install nome_pacchetto
