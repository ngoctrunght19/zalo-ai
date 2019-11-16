mkdir data
mkdir data/zalo

mkdir data/model
mkdir data/model/bert
mkdir data/model/output

mkdir data/glue
mkdir data/glue/MRPC

wget https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/train.zip
wget https://dl.challenge.zalo.ai/ZAC2019_VietnameseWikiQA/test.zip
unzip train.zip data/zalo
unzip test.zip data/zalo

pip install -r requirements.txt
python convert_data.py
