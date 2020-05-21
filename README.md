# bert-multi-gpu

###### tags: `README` `NCHC`
https://github.com/HaoyuHu/bert-multi-gpu
## 建立並連線 開發型容器
### ssh 
```
ssh u4839782@203.145.219.150 -p 51261
```
### filezilla
- 主機 `sftp://203.145.219.150`
- 使用者名稱 `u4839782`
- 密碼 
- 連接阜 `51261`

## BERT Finetune with custom data
### 環境安裝
```
python == 3.6
tensorflow >= 1.11.0 # CPU Version of TensorFlow.
tensorflow-gpu >= 1.11.0 # GPU version of TensorFlow.
```
### 預訓練模型下載
https://github.com/ymcui/Chinese-BERT-wwm
- BERT-wwm, Chinese
- BERT-base, Chinese (12-layer, 768-hidden, 12-heads, 110M parameters)

以TensorFlow版`BERT-wwm, Chinese`为例，下载完毕后对zip文件进行解压得到：
```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # 模型权重
    |- bert_model.meta      # 模型meta信息
    |- bert_model.index     # 模型index信息
    |- bert_config.json     # 模型参数
    |- vocab.txt            # 词表
```
其中`bert_config.json`和`vocab.txt`与谷歌原版`BERT-base, Chinese`完全一致

### 修改 run_classifier.py
- 自訂義 DataProcessor (二分類問題)
```python=225
class SentimentProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        train_df = pd.read_csv(file_path, encoding='utf-8')
        train_data = []
        for index, train in enumerate(train_df.values):
            guid = 'train-%d' % index
            text_a = tokenization.convert_to_unicode(str(train[0]))
            label = str(train[1])
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return train_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.csv')
        dev_df = pd.read_csv(file_path, encoding='utf-8')
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(dev[0]))
            label = str(dev[1])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path, encoding='utf-8')
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(test[0]))
            label = str(0) 
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return test_data

    def get_sentence_examples(self, questions):
        for index, data in enumerate(questions):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(data[0]))
            text_b = tokenization.convert_to_unicode(str(data[1]))
            label = str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        return ['0', '1']
```
- 自訂義 DataProcessor (三分類問題)
```python=271
class Classificaiton_3_Processor(DataProcessor):
    def get_train_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'train.csv')
        train_df = pd.read_csv(file_path, encoding='utf-8')
        train_data = []
        for index, train in enumerate(train_df.values):
            guid = 'train-%d' % index
            text_a = tokenization.convert_to_unicode(str(train[0]))
            text_b = tokenization.convert_to_unicode(str(train[1]))
            label = str(train[2])
            train_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return train_data

    def get_dev_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'dev.csv')
        dev_df = pd.read_csv(file_path, encoding='utf-8')
        dev_data = []
        for index, dev in enumerate(dev_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(dev[0]))
            text_b = tokenization.convert_to_unicode(str(dev[1]))
            label = str(dev[2])
            dev_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return dev_data

    def get_test_examples(self, data_dir):
        file_path = os.path.join(data_dir, 'test.csv')
        test_df = pd.read_csv(file_path, encoding='utf-8')
        test_data = []
        for index, test in enumerate(test_df.values):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(test[0]))
            text_b = tokenization.convert_to_unicode(str(test[1]))
            label = str(0) #str(test[2]) 
            test_data.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return test_data

    def get_sentence_examples(self, questions):
        for index, data in enumerate(questions):
            guid = 'test-%d' % index
            text_a = tokenization.convert_to_unicode(str(data[0]))
            text_b = tokenization.convert_to_unicode(str(data[1]))
            label = str(0)
            yield InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)

    def get_labels(self):
        return ['0', '1', '2']
```
- 添加 Processor
```python=764
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "classification_3": Classificaiton_3_Processor,
        "sentiment": SentimentProcessor,
    }
```

### 修改數據格式
- `train.csv` : 訓練集
- `dev.csv` : 驗證集
- `test.csv` : 測試集

```
paragraph, syn_paragraph, label
前國民黨主席洪秀柱今日下午抵達北京明日將與中共總書記習近平舉行會晤 ,國民黨前主席洪秀柱陳麒全攝國民黨前主席洪秀柱12日中午率團啟程前往大陸北京訪問 ,2
```

### 修改 train.sh

- `task_name`
- `data_dir`
- `output_dir`
- `max_seq_length`
- `train_batch_size`
- `learning_rate`
- `num_train_epochs`

```
python run_classifier.py \
  --task_name=classification_3 \
  --do_lower_case=true \
  --do_train=true \
  --do_eval=true \
  --do_predict=false \
  --save_for_serving=true \
  --data_dir=./data/classification/news_related_1541 \
  --vocab_file=./models/bert_pretrained/chinese_wwm_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=./models/bert_pretrained/chinese_wwm_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./models/bert_pretrained/chinese_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=10 \
  --use_gpu=true \
  --num_gpu_cores=8 \
  --use_fp16=true \
  --output_dir=./models/classification/news_related_1541_wwm_E-10
```