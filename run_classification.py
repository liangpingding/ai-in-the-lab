#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import sys
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd 
import datasets
import evaluate
import numpy as np
import wandb
import torch
import torch.nn.functional as F
from datasets import Value, load_dataset
from datasets import DatasetDict
from datasets import Dataset
import transformers
import json
import pprint
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import TrainerCallback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # 只让程序看到 GPU0
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
from sklearn.metrics import f1_score


if torch.cuda.is_available():
    print(f" GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU")
    
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    f1 = f1_score(labels, preds, average="macro")  
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

class UnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch):  
        self.unfreeze_epoch =unfreeze_epoch
        self.already_unfrozen = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch >= self.unfreeze_epoch and not self.already_unfrozen:
            model = kwargs["model"]
            for name, param in model.bert.named_parameters():
                if "encoder.layer.10" in name or "encoder.layer.11" in name or "encoder.layer.9" in name or "encoder.layer.8" in name:
                    param.requires_grad = True
            self.already_unfrozen = True
        return control


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    freeze_bert: bool = field(default=False, metadata={"help": "Whether to freeze bert parameters."})
    unfreeze_epoch: int = field(default=2, metadata={"help": "The epoch to unfreeze bert parameters."})
    early_stopping_patience: int = field(default=5, metadata={"help": "The patience to wait before early stopping."})
    nolabel_test: bool = field(default=False, metadata={"help": "Whether the test dataset has no labels."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "The delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

@dataclass
class wandbArguments:
    """Aruguments for wandb logging"""
    sweep_method: str = field(default='None')
    sweep_file: str = field(default='None')
    project_name: str = field(default='None')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    print('label list is {}'.format(';'.join(label_list)))
    return label_list


def wandb_train(num_labels,train_dataset,eval_dataset,model_args, data_args, training_args, wandb_args):
    with wandb.init():
        wandbconfig = wandb.config
        print("wandb config:------------------\n\n")
        print(wandbconfig)
        args = training_args
        args.learning_rate = wandbconfig.learning_rate
        args.num_train_epochs = wandbconfig.num_train_epochs

        set_seed(args.seed)
        # print("Seed value:", args.seed, type(args.seed))
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="text-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )


        model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
        tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]
            )            
        trainer.train()
        set_seed(training_args.seed)
        trainer.evaluate()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,wandbArguments))

    
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()
    
    send_example_telemetry("run_classification", model_args, data_args)

    set_seed(training_args.seed)
    ######################### Setup logging########################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


 
    ######################### process dataset ########################################
    if data_args.train_file.endswith(".csv"):
            print(data_args.train_file)
            train_df = pd.read_csv(data_args.train_file, usecols=[data_args.text_column_names, data_args.label_column_name])
            train_dataset = Dataset.from_pandas(train_df)
            if training_args.do_eval:
                eval_df = pd.read_csv(data_args.validation_file, usecols=[data_args.text_column_names, data_args.label_column_name])
                eval_dataset = Dataset.from_pandas(eval_df)
            if training_args.do_predict:
                
                if data_args.nolabel_test:
                    predict_df = pd.read_csv(data_args.test_file, usecols=[data_args.text_column_names])
                    predict_df[data_args.label_column_name]=-1
                else:
                    predict_df = pd.read_csv(data_args.test_file, usecols=[data_args.text_column_names, data_args.label_column_name])
                predict_dataset = Dataset.from_pandas(predict_df)

    raw_datasets = DatasetDict()   
    if train_dataset is not None:
        raw_datasets["train"] = train_dataset
    if training_args.do_eval and eval_dataset is not None:
        raw_datasets["validation"] = eval_dataset
    if training_args.do_predict and predict_dataset is not None:
        raw_datasets["test"] = predict_dataset
   
    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    label_list = get_label_list(raw_datasets, split="train")
    # if label is -1, we throw a warning and remove it from the label list
    for label in label_list:
        if label == -1:
            logger.warning("Label -1 found in label list, removing it.")
            label_list.remove(label)

    label_list.sort()
    num_labels = len(label_list)
    label_to_id = {v: i for i, v in enumerate(label_list)}
    print("label_to_id--------------------------------\n")
    print(label_to_id)

   ######################### Load model config and tokenizer ########################################
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    if data_args.freeze_bert:
        print('freeze bert parameters')
        for param in model.bert.parameters():
            param.requires_grad = False
        # for name, param in model.bert.named_parameters():
        #     if name.startswith("embeddings") or name.startswith("encoder.layer.0") or \
        #     name.startswith("encoder.layer.1") or name.startswith("encoder.layer.2") or \
        #     name.startswith("encoder.layer.3") or name.startswith("encoder.layer.4") or \
        #     name.startswith("encoder.layer.5") or name.startswith("encoder.layer.6") or \
        #     name.startswith("encoder.layer.7") or name.startswith("encoder.layer.8") :
        #         param.requires_grad = False
    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(examples["sentence"], padding="max_length", max_length=data_args.max_seq_length, truncation=True)
        # print('the output of {}'.format(label_to_id))
        if label_to_id is not None and (training_args.do_train or training_args.do_eval):
                result["label"] = [label_to_id[str(l)] if l is not None and l != -1 else -1 for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        for index in range(len(train_dataset)):
            if index<=3:
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
        else:
            eval_dataset = raw_datasets["validation"]

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]


    ###########################tune model parameters or train/test model####################
    if wandb_args.sweep_method=="grid":  
            with open(wandb_args.sweep_file) as f:
                    sweep_config = json.load(f)
            pprint.pprint(sweep_config)
            sweep_id=wandb.sweep(sweep_config,project=wandb_args.project_name)
            set_seed(training_args.seed)
            # import ipdb; ipdb.set_trace()
            wandb.agent(sweep_id,lambda: wandb_train(num_labels,train_dataset,eval_dataset,model_args, data_args, training_args, wandb_args))
            set_seed(training_args.seed)
    else:
        # Initialize our Trainer
        set_seed(training_args.seed)
        if data_args.freeze_bert:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience),UnfreezeCallback(data_args.unfreeze_epoch)]
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.early_stopping_patience)]
            )
        # Training
        if training_args.do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            set_seed(training_args.seed)
        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if training_args.do_predict:
            logger.info("*** Predict ***")
            # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
            if not data_args.nolabel_test:
                pred_output = trainer.predict(test_dataset=predict_dataset)
                metrics = dict(pred_output.metrics)  
                metrics["test_samples"] = len(predict_dataset)
                wandb.log(metrics)
                trainer.log_metrics("test", metrics)
                trainer.save_metrics("test", metrics)
            if "label" in predict_dataset.features:
                predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            logits = predictions  
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()  # shape: [N, C]
            predicted_indices = np.argmax(probs, axis=1)
            confidence_scores = np.max(probs, axis=1)


            rows = []
            for i, (pred_idx, conf, prob_row) in enumerate(zip(predicted_indices, confidence_scores, probs)):
                row = {
                    "index": i,
                    "prediction": label_list[pred_idx],
                    "confidence": conf
                }

                for j, label in enumerate(label_list):
                    row[f"prob_{label}"] = prob_row[j]
                rows.append(row)

            df = pd.DataFrame(rows)
            output_csv_file = os.path.join(training_args.output_dir, "predict_probs.csv")
            df.to_csv(output_csv_file, index=False)

            logger.info("Predict results saved at {}".format(output_csv_file))
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}






if __name__ == "__main__":
    wandb.login(key="")
    main()

