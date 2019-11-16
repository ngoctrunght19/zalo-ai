import json
import pandas as pd
from sklearn.model_selection import train_test_split

def convert_train():
  with open("data/zalo/train.json", encoding='utf-8') as f:
    data = json.load(f)

  examples = []
  for idx, example in enumerate(data):
    temp = {
      "label": example["label"]*1,
      "id_1": idx*2,
      "id_2": idx*2 + 1,
      "text": example["text"].replace("\r", "").replace("\n", ""),
      "question": example["question"].replace("\r", "").replace("\n", "")
    }
    examples.append(temp)

  df = pd.DataFrame(examples)
  train, test = train_test_split(df, test_size=0.2)
  train.to_csv("data/glue/MRPC/train.tsv", sep='\t', encoding='utf-8', index=False)
  test.to_csv("data/glue/MRPC/dev.tsv", sep='\t', encoding='utf-8', index=False)

def convert_test():
  with open("data/zalo/test.json", encoding='utf-8') as f:
    data = json.load(f)

  examples = []
  for example in data:
    temp = example.copy()
    del temp['paragraphs']
    for para in example["paragraphs"]:
      temp["text"] = para["text"].replace("\r", "").replace("\n", "")
      temp["pid"] = para["id"]
      examples.append(temp.copy())

  df = pd.DataFrame(examples)
  df.to_csv("data/glue/MRPC/test.tsv", sep='\t', encoding='utf-8', index=False)

convert_train()
convert_test()
