from datasets import load_dataset, load_metric, list_metrics
import pandas as pd
import evaluate
import numpy as np
from transformers import AutoTokenizer


def compute_metrics(eval_preds):
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    bleu = load_metric("sacrebleu")
    ter = load_metric("ter")
    rouge = evaluate.load("rouge")

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    ter_score = ter.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": bleu_score["score"]}
    result["ter"] = ter_score["score"]
    result["rouge1"] = rouge_score["rouge1"]
    result["rouge2"] = rouge_score["rouge2"]

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result
