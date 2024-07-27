import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# 读取CSV文件
data = pd.read_csv('distilbert_dataset.csv')

# 将关键词和内容分开
contents = data['Content'].tolist()
labels = data['Keywords'].tolist()

# 将标签转换为整数
label_to_id = {label: idx for idx, label in enumerate(set(labels))}
int_labels = [label_to_id[label] for label in labels]

# 创建训练和验证数据集（按72%训练集和28%测试集划分）
train_texts, val_texts, train_labels, val_labels = train_test_split(contents, int_labels, test_size=0.28, random_state=42)

# 初始化分词器
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# 分词和标记化函数
def tokenize(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# 将训练和验证集进行分词和标记化
train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

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
train_dataset = TextDataset(train_encodings, train_labels)
val_dataset = TextDataset(val_encodings, val_labels)

# 初始化模型
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(int_labels)))

# 训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=16,  # 训练批次大小
    per_device_eval_batch_size=16,   # 验证批次大小
    warmup_steps=500,                # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
)

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# 开始训练
trainer.train()

# 保存模型和分词器
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
