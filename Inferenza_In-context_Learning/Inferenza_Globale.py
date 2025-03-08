import json

with open('/path/UniQA/lora_ft_reduced_chat/it_test.json', 'r') as file:
    data = json.load(file)

instruction = "Sei Unipa-GPT, il chatbot e assistente virtuale dell'Università degli Studi di Palermo.\nRispondi in modo cordiale e colloquiale alle domande fornite.\nSe ricevi un saluto, rispondi salutando e presentandoti.\nSe ricevi una domanda per quanto riguarda l'Università degli Studi di Palermo, rispondi facendo affidamento sulla documentazione che ti è stata consegnata insieme alla domanda.\nSe non sai rispondere, chiedi scusa e suggerisci di consultare il sito dell'Università [https://www.unipa.it/], non inventare risposte.\nRispondi in italiano.\n"

from huggingface_hub import login

token = "token"
login(token=token)

def inference(model, tokenizer, messages, word, token, top_p, temperature, minerva):
    if minerva:
        prompt = messages
    else:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    for k,v in inputs.items():
        inputs[k] = v.to("cuda:1")
    outputs = model.generate(**inputs, max_new_tokens=token, do_sample=True, top_p=top_p, temperature=temperature)
    results = tokenizer.batch_decode(outputs)[0]

    if word in results:
        indice = results.find(word)
        return(results[indice + len(word):])
    else:
        return(results)
    
import evaluate

def evaluate_metrics(references, predictions, model_id):
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)

    rouge = evaluate.load('rouge')
    rouge_score = rouge.compute(predictions=predictions,references=references)
    
    bertscore = evaluate.load("bertscore")
    bert_score = bertscore.compute(predictions=predictions, references=references, lang="it", device="cpu")
    
    meteor = evaluate.load('meteor')
    meteor_score = meteor.compute(predictions=predictions, references=references)
    
    perplexity = evaluate.load("perplexity", module_type="metric")
    perplexity_score = perplexity.compute(predictions=predictions, model_id=model_id, device="cpu")
    
    perplexity_z_score = perplexity.compute(predictions=predictions, model_id="HuggingFaceH4/zephyr-7b-beta", device="cpu")
    
    return bleu_score, rouge_score, bert_score, meteor_score, perplexity_score, perplexity_z_score

def mean(list):
    mean = sum(list)/len(list)
    
    return mean

def inferences(model, tokenizer, data, word, model_id, token = 512, top_p = 0.9, temperature = 0.6, gemma = False, minerva = False):
    output_pairs = {'output': [], 'reference': []}

    for message in data:
        if gemma:
            messages = [
                {"role": "user", "content": "Ciao"},
                {"role": "assistant", "content": instruction},
                {"role": "user", "content": message['input']},
                {"role": "assistant", "content": ""}
            ]
        elif minerva:
            messages = """  """ + instruction + """ 

            ### Istruzione: """ + message['input'] + """

            ### Input:
        
            ### Risposta:"""
        else:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": message['input']}
            ]

        result = inference(model, tokenizer, messages, word, token, top_p, temperature, minerva)
        
        output_pairs['output'].append(result)
        output_pairs['reference'].append(message['output'])
        
        with open('output_metric_step.json', 'w') as file:
            json.dump(output_pairs, file, indent=4)
    
    bleu_score, rouge_score, bert_score, meteor_score, perplexity_score, perplexity_z_score = evaluate_metrics(output_pairs['reference'], output_pairs['output'], model_id)
    
    print(f"BLEU score: {round(bleu_score['bleu'], 2)}")
    print(f"ROUGE score: {rouge_score}")
    print(f"BERT_PRECISION score: {round(mean(bert_score['precision']), 2)}")
    print(f"BERT_RECALL score: {round(mean(bert_score['recall']), 2)}")
    print(f"BERT_F1 score: {round(mean(bert_score['f1']), 2)}")
    print(f"METEOR score: {round(meteor_score['meteor'], 2)}")
    print(f"PERPLEXITY score: {round(perplexity_score['mean_perplexity'], 2)}")
    print(f"PERPLEXITY-Z score: {round(perplexity_z_score['mean_perplexity'], 2)}")
    
    output_metric = {'output_pairs': output_pairs,
                     'bleu_score': bleu_score,
                     'rouge_score': rouge_score,
                     'bert_score': bert_score,
                     'meteor_score': meteor_score,
                     'perplexity_score': perplexity_score,
                     'perplexity_z_score': perplexity_z_score
                    }
    
    try:
        with open('output_metric.json', 'r') as file:
            output_metric_json = json.load(file)
    except FileNotFoundError:
        output_metric_json = []
    except json.JSONDecodeError:
        output_metric_json = []

    output_metric_json.append(output_metric)
    
    with open('output_metric.json', 'w') as file:
        json.dump(output_metric_json, file, indent=4)
        
def inferences_one_shot(model, tokenizer, data, word, model_id, token = 512, top_p = 0.9, temperature = 0.6, gemma = False, minerva = False, shot = False):
    output_pairs = {'output': [], 'reference': []}

    for message in data:
        if gemma:
            messages = [ 
                {"role": "user", "content": "Ciao"},
                {"role": "assistant", "content": instruction},
                {"role": "user", "content": data[3]['input']},
                {"role": "assistant", "content": data[3]['output']},
                {"role": "user", "content": message['input']},
                {"role": "assistant", "content": ""}
            ]
        elif minerva:
            messages = """  """ + instruction + """ 
            
            ### Istruzione: """ + data[3]['input'] + """

            ### Input: 
        
            ### Risposta: """ + data[3]['output'] + """
            
            ### Istruzione: """ + message['input'] + """

            ### Input:
        
            ### Risposta Generata:"""
        elif shot:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": data[3]['input']},
                {"role": "assistant_shot", "content": data[3]['output']},
                {"role": "user", "content": message['input']}
            ]
        else:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": data[3]['input']},
                {"role": "assistant", "content": data[3]['output']},
                {"role": "user", "content": message['input'] + "Risposta:\n"}
            ]

        result = inference(model, tokenizer, messages, word, token, top_p, temperature, minerva)
        
        output_pairs['output'].append(result)
        output_pairs['reference'].append(message['output'])
        
        with open('output_metric_one_shot.json', 'w') as file:
            json.dump(output_pairs, file, indent=4)

    bleu_score, rouge_score, bert_score, meteor_score, perplexity_score, perplexity_z_score = evaluate_metrics(output_pairs['reference'], output_pairs['output'], model_id)
    
    print(f"BLEU score: {round(bleu_score['bleu'], 2)}")
    print(f"ROUGE score: {rouge_score}")
    print(f"BERT_PRECISION score: {round(mean(bert_score['precision']), 2)}")
    print(f"BERT_RECALL score: {round(mean(bert_score['recall']), 2)}")
    print(f"BERT_F1 score: {round(mean(bert_score['f1']), 2)}")
    print(f"METEOR score: {round(meteor_score['meteor'], 2)}")
    print(f"PERPLEXITY score: {round(perplexity_score['mean_perplexity'], 2)}")
    print(f"PERPLEXITY-Z score: {round(perplexity_z_score['mean_perplexity'], 2)}")
    
    output_metric = {'output_pairs': output_pairs,
                     'bleu_score': bleu_score,
                     'rouge_score': rouge_score,
                     'bert_score': bert_score,
                     'meteor_score': meteor_score,
                     'perplexity_score': perplexity_score,
                     'perplexity_z_score': perplexity_z_score
                    }
    
    try:
        with open('output_metric_total_one_shot.json', 'r') as file:
            output_metric_json = json.load(file)
    except FileNotFoundError:
        output_metric_json = []
    except json.JSONDecodeError:
        output_metric_json = []

    output_metric_json.append(output_metric)
    
    with open('output_metric_total_one_shot.json', 'w') as file:
        json.dump(output_metric_json, file, indent=4)
        
def inferences_few_shot(model, tokenizer, data, word, model_id, token = 512, top_p = 0.9, temperature = 0.6, gemma = False, minerva = False, shot = False):
    output_pairs = {'output': [], 'reference': []}

    for message in data:
        if gemma:
            messages = [ 
                {"role": "user", "content": "Ciao"},
                {"role": "assistant", "content": instruction},
                {"role": "user", "content": data[3]['input']},
                {"role": "assistant", "content": data[3]['output']},
                {"role": "user", "content": data[4]['input']},
                {"role": "assistant", "content": data[4]['output']},
                {"role": "user", "content": message['input']},
                {"role": "assistant", "content": ""}
            ]
        elif minerva:
            messages = """  """ + instruction + """ 
            
            ### Istruzione: """ + data[3]['input'] + """

            ### Input: 
        
            ### Risposta: """ + data[3]['output'] + """
            
            ### Istruzione: """ + data[4]['input'] + """

            ### Input: 
        
            ### Risposta: """ + data[4]['output'] + """
            
            ### Istruzione: """ + message['input'] + """

            ### Input:
        
            ### Risposta Generata:"""
        elif shot:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": data[3]['input']},
                {"role": "assistant_shot", "content": data[3]['output']},
                {"role": "user", "content": data[4]['input']},
                {"role": "assistant_shot", "content": data[4]['output']},
                {"role": "user", "content": message['input']}
            ]
        else:
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": data[3]['input']},
                {"role": "assistant", "content": data[3]['output']},
                {"role": "user", "content": data[4]['input']},
                {"role": "assistant", "content": data[4]['output']},
                {"role": "user", "content": message['input'] + "Risposta:\n"}
            ]

        result = inference(model, tokenizer, messages, word, token, top_p, temperature, minerva)
        
        output_pairs['output'].append(result)
        output_pairs['reference'].append(message['output'])
        
        with open('output_metric_few_shot.json', 'w') as file:
            json.dump(output_pairs, file, indent=4)
    
    bleu_score, rouge_score, bert_score, meteor_score, perplexity_score, perplexity_z_score = evaluate_metrics(output_pairs['reference'], output_pairs['output'], model_id)
    
    print(f"BLEU score: {round(bleu_score['bleu'], 2)}")
    print(f"ROUGE score: {rouge_score}")
    print(f"BERT_PRECISION score: {round(mean(bert_score['precision']), 2)}")
    print(f"BERT_RECALL score: {round(mean(bert_score['recall']), 2)}")
    print(f"BERT_F1 score: {round(mean(bert_score['f1']), 2)}")
    print(f"METEOR score: {round(meteor_score['meteor'], 2)}")
    print(f"PERPLEXITY score: {round(perplexity_score['mean_perplexity'], 2)}")
    print(f"PERPLEXITY-Z score: {round(perplexity_z_score['mean_perplexity'], 2)}")
    
    output_metric = {'output_pairs': output_pairs,
                     'bleu_score': bleu_score,
                     'rouge_score': rouge_score,
                     'bert_score': bert_score,
                     'meteor_score': meteor_score,
                     'perplexity_score': perplexity_score,
                     'perplexity_z_score': perplexity_z_score
                    }
    
    try:
        with open('output_metric_total_few_shot.json', 'r') as file:
            output_metric_json = json.load(file)
    except FileNotFoundError:
        output_metric_json = []
    except json.JSONDecodeError:
        output_metric_json = []

    output_metric_json.append(output_metric)
    
    with open('output_metric_total_few_shot.json', 'w') as file:
        json.dump(output_metric_json, file, indent=4)
              
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "model_id"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

word = "word"
inferences(model, tokenizer, data, word, model_id)
#word = "word_shot_learning"
inferences_one_shot(model, tokenizer, data, word, model_id, shot=True) #inferences_one_shot(model, tokenizer, data, word, model_id)
inferences_few_shot(model, tokenizer, data, word, model_id, shot=True) #inferences_few_shot(model, tokenizer, data, word, model_id)
