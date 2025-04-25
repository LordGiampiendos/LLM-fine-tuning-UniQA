from ragas.metrics import Faithfulness, AnswerCorrectness
from ragas import evaluate
import nest_asyncio
import os
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Necessario per il calcolo delle metriche
nest_asyncio.apply()

# Token e Modello Scelto
os.environ["OPENAI_API_KEY"] = "token"
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# Indici delle Righe da Considerare nel Calcolo
index = [0, 1, 1156, 1158, 1162, 1168, 1170, 1184, 1238, 1241]

import json
from datasets import Dataset 

# Carica i dataset di valutazione
with open('/home/gbarbaro/Test/Inferenza_Fine_Tuning/Risultati_FT/output_metric_LLAMA_3.3_FT.json', 'r') as file:
    data = json.load(file)

j=0
output = [data[j]['output_pairs']['output'][i] for i in index] 
reference = [data[j]['output_pairs']['reference'][i] for i in index]

# Carica il dataset di test 
with open('/home/gbarbaro/UniQA/lora_ft_reduced_chat/it_test.json', 'r') as file:
    data_input = json.load(file)

user_input = [data_input[i]["question"] for i in index]

retrieved_contexts = []

for i in index:
    if "Documents" in data_input[i]["input"]: 
        indice = data_input[i]["input"].find("Documents") 
        retrieved_contexts.append(data_input[i]["input"][indice + len("Documents:\n"):].strip().split('\n')) 
    else: retrieved_contexts.append([])

# Converti tutto in un formato compatibile con Hugging Face Dataset 
dataset_dict = {"user_input": user_input, "reference": reference, "response": output, "retrieved_contexts": retrieved_contexts} 
hf_dataset = Dataset.from_dict(dataset_dict)

# Definizione Metriche
metrics = [
    AnswerCorrectness(llm = evaluator_llm),
    Faithfulness(llm = evaluator_llm),
]

# Calcolo e Stampa
results = evaluate(dataset=hf_dataset, metrics=metrics, raise_exceptions=False)

print(results)

# Salvataggio
try:
    with open('output_metric_RAG.json', 'r') as file:
        output_metric_json = json.load(file)
except FileNotFoundError:
    output_metric_json = []
except json.JSONDecodeError:
    output_metric_json = []

output_metric_json.append({"ac": results["answer_correctness"], "f": results["faithfulness"]})
    
with open('output_metric_RAG.json', 'w') as file:
    json.dump(output_metric_json, file, indent=4)
