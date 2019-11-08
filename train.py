import os
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf

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

train_df = pd.read_csv("data/train/train.csv", sep="|", encoding="utf-8", engine="python", dtype=str)
train_df = train_df.sample(2000)

train, test = train_test_split(train_df, test_size = 0.1, random_state=42)
train_texts, train_questions, train_labels = train.text.values, train.question.values, train.label.values # 1*(train.label.values == "True")
test_texts, test_questions, test_labels = test.text.values, test.question.values, train.label.values

def create_examples(texts_a, texts_b, labels=None):
    """Generate data for the BERT model"""

    examples = []
    if labels is not None:
        for text_a, text_b, label in zip(texts_a, texts_b, labels):
            examples.append(run_classifier.InputExample(guid="train", text_a=text_a, text_b=text_b, label=label))
    else:
        for text_a, text_b in zip(texts_a, texts_b):
            examples.append(run_classifier.InputExample(guid="test", text_a=text_a, text_b=text_b))

    return examples

# Model Hyper Parameters
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
MAX_SEQ_LENGTH = 128
# Model configs
SAVE_SUMMARY_STEPS = 200
SAVE_CHECKPOINTS_STEPS = 1000 #if you wish to finetune a model on a larger dataset, use larger interval
# each checpoint weights about 1,5gb
ITERATIONS_PER_LOOP = 1000
NUM_TPU_CORES = 8
VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
DO_LOWER_CASE = BERT_MODEL.startswith('uncased')

label_list = ['False', 'True']
tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

train_examples = create_examples(train_texts, test_questions, train_labels)

tpu_cluster_resolver = None
run_config = tf.contrib.tpu.RunConfig(
    model_dir=OUTPUT_DIR,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    tpu_config=tf.contrib.tpu.TPUConfig(
        iterations_per_loop=ITERATIONS_PER_LOOP,
        num_shards=NUM_TPU_CORES,
        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))

num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

model_fn = run_classifier.model_fn_builder(
    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),
    num_labels=len(label_list),
    init_checkpoint=INIT_CHECKPOINT,
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available
    use_one_hot_embeddings=True)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available
    model_fn=model_fn,
    config=run_config,
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE)

# Train the model.
print('Please wait...')
train_features = run_classifier.convert_examples_to_features(
    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
print('***** Started training at {} *****'.format(datetime.datetime.now()))
print('  Num examples = {}'.format(len(train_examples)))
print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
tf.logging.info("  Num steps = %d", num_train_steps)
train_input_fn = run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('***** Finished training at {} *****'.format(datetime.datetime.now()))
