# LLM-fine-tuning-UnipaQA

This repository contains the code for fine-tuning language models using the UniQA dataset. The UniQA dataset is available on GitHub at the following link: https://github.com/CHILab1/UniQA/tree/main.

@inproceedings{siragusa2024uniqa,
	author = {Siragusa, Irene and Pirrone, Roberto},
	title={UniQA: an Italian and English Question-Answering Data Set Based on Educational Documents},
	booktitle={Proceedings of the Eighth Workshop on Natural Language for Artificial Intelligence (NL4AI 2024) co-located with 23th International Conference of the Italian Association for Artificial Intelligence (AI* IA 2024)},
	year = {2024},
	address= {Bolzano, Italy}
}

## Repository Structure

The repository is organized as follows:

- `Fine_Tuning/`: Contains the scripts and code necessary for fine-tuning the models.
- `Inferenza_In-context_Learning/`: Includes the scripts for inference using in-context learning and the results obtained in terms of generated responses and calculated metrics.
- `Metriche-LLM-10-GPT/`: Contains the scripts for evaluation with AI-based evaluation metrics (RAGAS/gpt-4o-mini).

## Requirements

Ensure that the following Python packages are installed:

- jupyter
- torch
- transformers (a modified trainer is present in the Fine_Tuning/Modified folder, where an additional metric for the training process (BERT F1 score) has been added)
- evaluate
- huggingface_hub
- trl
- datasets
- peft
- ragas
- nest_asyncio
- langchain_openai
- matplotlib
- numpy

You can install the necessary requirements using `pip`:

```bash
pip install package_name
