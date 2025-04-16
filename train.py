import os
import json
import logging
from functools import partial
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass, field

import evaluate

import datasets
from datasets import DatasetDict, load_dataset

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    default_data_collator,
    set_seed,
)
from preprocess import build_tokenizer, load_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize(example):
    return tokenizer(example["text"])

def group_texts(examples):
    # concat input_ids
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size

    # split block_size
    result = {
        k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
        for k in concatenated
    }

    # prepare labels
    result["labels"] = result["input_ids"].copy()
    return result

def main():
    set_seed(training_args.seed)
  
    # load dataset
    ds = load_dataset("datablations/c4-filter-small", split="train")
    ds = ds.select_columns(["text"])
    ds = ds.train_test_split(test_size=0.1)
    logger.info(f"Dataset loaded: {raw_dataset}")

    # load pretrained model and tokenizer
    tokenizer_file = build_tokenizer(ds)
    tokenizer = load_tokenizer(tokenizer_file)
    tokenized_ds = ds.map(
        tokenize, remove_columns=["text"], batched=True, num_proc=4
    )
    lm_ds = tokenized_ds.map(group_texts, batched=True, num_proc=4)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir="gpt-small-c4",
        logging_dir="logs",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        logging_steps=1000,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator
    )
  
    trainer.train()

if __name__ == "__main__":
    main()
