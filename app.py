# app.py
import torch
import torch.nn as nn
from transformers import BertTokenizer
import gradio as gr
from utils import clean_text

# ✅ تعریف کلاس مدل BERT + BiLSTM
class BERTBiLSTM(nn.Module):
    def __init__(self, hidden_size=128, num_classes=2):
        super(BERTBiLSTM, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # Freeze BERT during inference
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(bert_output)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)

# ✅ بارگذاری مدل و توکنایزر
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BERTBiLSTM()
model.load_state_dict(torch.load("model/pytorch_model.bin", map_location="cpu"))
model.eval()

# ✅ تابع پیش‌بینی
def predict_sentiment(text):
    cleaned = clean_text(text)
    encoded = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        pred = torch.argmax(output, dim=1).item()

    return "🟢 Positive" if pred == 1 else "🔴 Negative"

# ✅ رابط Gradio
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a movie review..."),
    outputs="text",
    title="🎬 BERT + BiLSTM Sentiment Classifier",
    description="Enter a review and see whether it's classified as Positive or Negative using a BERT+BiLSTM model."
)

if __name__ == "__main__":
    iface.launch()
