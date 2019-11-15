import json
import pandas as pd

def convert_train():
  with open("data/zalo/train.json", encoding='utf-8') as f:
    data = json.load(f)

  examples = []
  for idx, example in enumerate(data):
    temp = {
      "label": example["label"]*1,
      "id_1": idx*2,
      "id_2": idx*2 + 1,
      "text": example["text"].rstrip(),
      "question": example["question"].rstrip()
    }
    examples.append(temp)

  df = pd.DataFrame(examples)
  df.to_csv("data/glue/MRPC/train.tsv", sep='\t', encoding='utf-8', index=False)

def convert_test():
  with open("data/zalo/test.json", encoding='utf-8') as f:
    data = json.load(f)

  examples = []
  for example in data:
    temp = example.copy()
    del temp['paragraphs']
    for para in example["paragraphs"]:
      temp["text"] = para["text"].rstrip()
      temp["pid"] = para["id"]
      examples.append(temp.copy())

  df = pd.DataFrame(examples)
  df.to_csv("data/glue/MRPC/test.tsv", sep='\t', encoding='utf-8', index=False)

convert_train()
convert_test()
