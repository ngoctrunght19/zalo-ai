import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from bert import modeling, optimization, run_classifier, tokenization

BERT_MODEL = "multi_cased_L-12_H-768_A-12"
BERT_PRETRAINED_DIR = f"data/model/bert/multi_cased_L-12_H-768_A-12"
OUTPUT_DIR = f"data/model/output"
print(f"***** Model output directory: {OUTPUT_DIR} *****")
print(f"***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****")

VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, "vocab.txt")
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, "bert_config.json")
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, "bert_model.ckpt")
DO_LOWER_CASE = BERT_MODEL.startswith("uncased")

train_df = pd.read_csv("data/train/train.csv", sep="|", encoding="utf-8", engine="python")
train_df = train_df.sample(2000)

train, test = train_test_split(train_df, test_size = 0.1, random_state=42)
train_texts, train_questions, train_labels = train.text.values, train.question.values, 1*(train.label.values == "True")

test_texts, test_questions, test_labels = test.text.values, test.question.values, 1*(test.label.values == "True")

print(test_labels)

tokenizer = tokenization.FullTokenizer(
    vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE
)
