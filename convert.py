import json
import pandas as pd

def convert_train():
  with open("data/train/train.json") as f:
    data = json.load(f)

  df = pd.DataFrame(data)
  df.to_csv("data/train/train.csv", sep='|', encoding='utf-8')

def convert_test():
  with open("data/test/test.json") as f:
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
  df.to_csv("data/test/test.csv", sep='|', encoding='utf-8', index=False)

convert_train()
convert_test()
