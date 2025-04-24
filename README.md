# LLM-fine-tuning-UnipaQA

This repository contains the code for fine-tuning language models using the UniQA dataset. The UniQA dataset is available on GitHub at the following link: https://github.com/CHILab1/UniQA/tree/main.

```
@inproceedings{siragusa2024uniqa,
	author = {Siragusa, Irene and Pirrone, Roberto},
	title={UniQA: an Italian and English Question-Answering Data Set Based on Educational Documents},
	booktitle={Proceedings of the Eighth Workshop on Natural Language for Artificial Intelligence (NL4AI 2024) co-located with 23th International Conference of the Italian Association for Artificial Intelligence (AI* IA 2024)},
	year = {2024},
	address= {Bolzano, Italy}
}
```

## Repository Structure

The repository is organized as follows:

- `Dataset/`: Contains the italian dataset train-test split.
- `Fine_Tuning/`: Contains the scripts and code necessary for fine-tuning the models.
- `Inferenza_In-context_Learning/`: Includes the scripts for inference using in-context learning and the results obtained in terms of generated responses and calculated metrics.
- `MMLU-IT/`: Contains the results of MMLU-IT metric.
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
```

You could also create a virtual environment in python by venv or conda.

### Venv
```bash
python -m venv env_name

Windows
.\env_name\Scripts\activate

Linux
source env_name/bin/activate
```

### Conda
```bash
conda create --name myenv python=3.9

conda activate myenv
```

I used two main server: CHILab server and LEONARDO by CINECA.

### LEONARDO
If you want to execute this code in LEONARDO with VENV, you should follow this steps for the initialization.

```
module load python
module load cuda

... create VENV ...
```

In the job script:
```
...
#SBATCH --gres=gpu:4
...
module load python
module load cuda
source /leonardo_work/IscrC_DOC-VLM/gbarbaro/my_venv/bin/activate

python /leonardo_work/path/Codice/Python.py

or for MMLU-IT evaluation

lm-eval --model hf --model_args pretrained=/leonardo_work/path/best_model --tasks m_mmlu_it --num_fewshot 5 --device cuda --batch_size 1 --log_samples --output_path results
```

## MMLU-IT Evaluation

For the MMLU-IT evaluation you should use the Language Model Evaluation Harness at the following link: https://github.com/EleutherAI/lm-evaluation-harness.

```
@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}
```

## Other Things

For other information consult the README.md in every folder.
