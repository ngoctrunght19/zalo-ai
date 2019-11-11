import json
import pandas as pd

def convert_train():
  with open("data/zalo/train.json") as f:
    data = json.load(f)

  examples = []
  for idx, example in enumerate(data):
    temp = {
      "label": example["label"]*1,
      "id_1": idx*2,
      "id_2": idx*2 + 1,
      "text": example["text"],
      "question": example["question"]
    }
    examples.append(temp)

  df = pd.DataFrame(examples)
  df.to_csv("data/glue/zalo/train.tsv", sep='\t', encoding='utf-8')

def convert_test():
  with open("data/zalo/test.json") as f:
    data = json.load(f)

  examples = []
  for example in data:
    temp = example.copy()
    del temp['paragraphs']
    for para in example["paragraphs"]:
      temp["text"] = para["text"]
      temp["pid"] = para["id"]
      examples.append(temp.copy())

  df = pd.DataFrame(examples)
  df.to_csv("data/glue/zalo/test.tsv", sep='\t', encoding='utf-8')

convert_train()
convert_test()
