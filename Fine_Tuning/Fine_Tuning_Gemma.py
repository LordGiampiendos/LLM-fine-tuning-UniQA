from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig

# Token Hugging Face
token = "token"
model_name_or_path = "/path/LLM/gemma2-9b-inst"

# PEFT Config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM",
)

# Model Import
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name_or_path,
    token=token,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

import json

# Train Dataset
with open('/path/Dataset/it_train.json', 'r') as file:
    data = json.load(file)
    
from sklearn.model_selection import train_test_split

# Seed e Split Methods
seed = 36
train_data, validation_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=seed)

# Datasets Creation
from datasets import Dataset, DatasetDict

def convert_to_dict(data): 
    result = {k: [dic[k] for dic in data] for k in data[0]} 
    return result 

# Train Dateset
dataset_dict = convert_to_dict(train_data)
dataset_train = Dataset.from_dict(dataset_dict)

# Validation Dataset
dataset_dict = convert_to_dict(validation_data)
dataset_validation = Dataset.from_dict(dataset_dict)

instruction = "Sei Unipa-GPT, il chatbot e assistente virtuale dell'Università degli Studi di Palermo.\nRispondi in modo cordiale e colloquiale alle domande fornite.\nSe ricevi un saluto, rispondi salutando e presentandoti.\nSe ricevi una domanda per quanto riguarda l'Università degli Studi di Palermo, rispondi facendo affidamento sulla documentazione che ti è stata consegnata insieme alla domanda.\nSe non sai rispondere, chiedi scusa e suggerisci di consultare il sito dell'Università [https://www.unipa.it/], non inventare risposte.\nRispondi in italiano.\n"

# Chat Template
def format_chat_template(row):
    row_json = [{"role": "user", "content": "Ciao"},
                {"role": "assistant", "content": instruction},
                {"role": "user", "content": row['input']},
                {"role": "assistant", "content": row["output"]}]
        
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False, truncation=True)
    return row


tokenized_datasets_train = dataset_train.map(
    format_chat_template,
    num_proc= 4,
)

tokenized_datasets_validation = dataset_validation.map(
    format_chat_template,
    num_proc= 4,
)

tokenized_datasets_train = tokenized_datasets_train.remove_columns(['input', 'question', 'context', 'instruction'])
tokenized_datasets_validation = tokenized_datasets_validation.remove_columns(['input', 'question', 'context', 'instruction'])
tokenized_datasets_train = tokenized_datasets_train.rename_column('output', 'labels')
tokenized_datasets_validation = tokenized_datasets_validation.rename_column('output', 'labels')

# Global Dataset 
tokenized_datasets = DatasetDict({"train": tokenized_datasets_train, "validation": tokenized_datasets_validation})

from trl import SFTTrainer, SFTConfig

# Training Args
training_args = SFTConfig(
    output_dir="output_dir",
    num_train_epochs=30,
    per_device_train_batch_size=2,   
    per_device_eval_batch_size=10,
    eval_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    do_eval=True,
    do_train=True,
    dataset_text_field="text",
    max_seq_length=512,
    load_best_model_at_end=True, 
    metric_for_best_model="eval_loss",
    eval_on_start=True,
    packing=True
) 

# Training with SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    args=training_args,
    peft_config=peft_config
)

trainer.train()
