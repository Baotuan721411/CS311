# Team CS311.Q12 At SemEval-2026 Task 12

This repository contains our system for **SemEval-2026 Task 12 – Abductive Event Reasoning**, focusing on information retrieval and large language model–based reasoning.

This work is brought to you by:

- **Ngô Hồng Vinh** (24522016) — Student at UIT, VNU-HCM  
- **Trương Đoàn Bảo Tuấn** (24521946) — Student at UIT, VNU-HCM  
- **Nguyễn Khôi Nguyên** (24521193) — Student at UIT, VNU-HCM

---

## Installation

To create a conda environment and install all the necessary libraries:

```bash
conda create -n CS311 python=3.12
conda activate CS311
pip install -r requirements.txt
```

## Quick Start
### Project Structure
```
project-root/
│
├── config/
│   ├── RetrieverConfig.json
│   └── TrainerConfig.json
│
├── data/
│   ├── dev_data/
│   │   ├── docs.json
│   │   └── questions.jsonl
│   │
│   ├── sample_data/
│   │   ├── docs.json
│   │   └── questions.jsonl
│   │
│   ├── test_data/
│   │   ├── docs.json
│   │   └── questions.jsonl
│   │
│   └── train_data/
│       ├── docs.json
│       └── questions.jsonl
│
├── processed/
│   ├── dev_data_final.jsonl
│   ├── test_data_final.jsonl
│   └── train_data_final.jsonl
│
├── prompting_technique/
│   └── ZeroShot.txt
│
├── src/
│   ├── Retriever.py
│   ├── Trainer.py
│   ├── eval.py
│   ├── test.py
│   └── utils.py
│    
│
├── main.py
├── test.py
├── requirements.txt
└── README.md
```
## Run Pipeline

If you want to fine-tune your own models, please create a configuration file with the same format as `TrainerConfig.json`.

### Step 1: Retrieve documents and build training data

```bash
python main.py --stage retrieve --retriever SBertRetriever
```
This step performs document retrieval and constructs the processed data used for training. You can replace SBertRetriever with BM25Retriever or HybridRetriever.

### Step 2: Fine-tune the model
```bash
python main.py --stage train --lora_config config/TrainerConfig.json
```
This step starts the fine-tuning process using the specified LoRA configuration. Make sure the config path is correct if you custom your own config.

### Step 3: Evaluate the model
```bash
python main.py --stage eval --lora_config config/TrainerConfig.json
```
This step evaluates the fine-tuned model on the development data and exports
the evaluation results to a PDF file.

## Note
The `test.py` is just a quick implementation for submission.
