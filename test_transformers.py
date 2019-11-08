import tensorflow as tf
from transformers import BertConfig, BertTokenizer

MODEL = "bert-base-multilingual-cased"
config = BertConfig.from_pretrained(MODEL)
tokenizer = BertTokenizer.from_pretrained(MODEL)
# model = BertForSequenceClassification.from_pretrained(MODEL)
print(tokenizer.tokenize("Sẽ có nhiều xáo trộn lớn trên BXH sau các trận đấu ở vòng 12 Ngoại hạng Anh"))
