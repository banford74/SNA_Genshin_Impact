import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=3)  # 3 for sentiment classes: positive, negative, neutral

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['Post']
        label = self.data.iloc[idx]['Sentiment']

        # Tokenize and encode the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Read Excel data including both 'Post' and 'Sentiment' columns
excel_file_path = "fine-tune.xls"
data = pd.read_excel(excel_file_path)

# Define training dataset
train_dataset = CustomDataset(data, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
    output_dir='./output_model_dir',  # Define output_model_dir directory for saving fine-tuned model
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
output_model_dir = "./fine_tuned_model"
trainer.save_model(output_model_dir)

print("Model fine-tuning completed. Fine-tuned model saved to:", output_model_dir)
