import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from openai import OpenAI

client = OpenAI(api_key="sk-660e24e4045f4defa15ab6d64a0b4c27", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese", num_labels=3) # 3 for sentiment classes: positive, negative, neutral

# Tokenize and prepare input
def prepare_input(text):
    if not isinstance(text, str):
        text = str(text)  # Convert to string if not already
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    return encoded_input

# Perform sentiment analysis
def sentiment_analysis(text):
    input_data = prepare_input(text)
    with torch.no_grad():
        output = model(**input_data)
    scores = output.logits
    predicted_class = torch.argmax(scores).item()
    if predicted_class == 0:
        return "Positive"
    elif predicted_class == 1:
        return "Negative"
    else:
        return "Neutral"

# Read Excel data for only the 'post' column
excel_file_path = "genshin_liyue.xls"
data = pd.read_excel(excel_file_path, usecols=['Post'])

# Apply sentiment analysis to each 'post' and create a new column 'Sentiment'
data['Sentiment'] = data['Post'].apply(sentiment_analysis)

# Save the results back to Excel
output_excel_file_path = "Liyue_SA_results.xlsx"
data.to_excel(output_excel_file_path, index=False)

print("Sentiment analysis completed. Results saved to:", output_excel_file_path)
