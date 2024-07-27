from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd

# 加载模型和分词器
model = DistilBertForSequenceClassification.from_pretrained('./model')
tokenizer = DistilBertTokenizer.from_pretrained('./model')

# 加载验证数据
data = pd.read_csv('distilbert_dataset.csv')

# 将关键词和内容分开
contents = data['Content'].tolist()
labels = data['Keywords'].tolist()

# 将标签转换为整数
label_to_id = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_id[label] for label in labels]

# 分词和标记化函数
def tokenize(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# 将验证集进行分词和标记化
val_encodings = tokenize(contents)

# 创建PyTorch数据集类
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# 创建数据集
val_dataset = TextDataset(val_encodings, int_labels)

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    per_device_eval_batch_size=16,   # 验证批次大小
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=val_dataset,
)

# 评估模型
eval_results = trainer.evaluate()
print(eval_results)
