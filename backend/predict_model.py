from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# 加载模型和分词器
tokenizer = DistilBertTokenizer.from_pretrained('./model')
model = DistilBertForSequenceClassification.from_pretrained('./model')

# 进行预测
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# 示例
text = "你想要预测的文本"
prediction = predict(text)
print(f"Prediction: {prediction}")
