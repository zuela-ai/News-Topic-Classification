# News-Topic-Classification Model Card

## 📰 Introduction

Welcome to the News-Topic-Classification repository, a collection of language-specific models for classifying news topics. Each model is fine-tuned for a specific language, including Kimbundu, Umbundu, Kikongo, Chokwe, and Lua. These models are derived from the AngoBERTa architecture and were trained on the SIB-200 dataset, which can be found here.

## 🌍 Language Models

Kimbundu:
Model: Hugging Face - Kimbundu News Category Classification
Umbundu:
Model: Hugging Face - Umbundu News Category Classification
Kikongo:
Model: Hugging Face - Kikongo News Category Classification
Chokwe:
Model: Hugging Face - Chokwe News Category Classification
Lua:
Model: Hugging Face - Lua News Category Classification
## 🚀 Training Details

Base Model: AngoBERTa
Dataset: SIB-200 Dataset
## 📊 Evaluation Metrics (Sample)

Epochs: 3.0
Training Loss: [Provide training loss]
Evaluation Loss: [Provide evaluation loss]
Accuracy: [Provide accuracy]
Precision: [Provide precision]
Recall: [Provide recall]
F1 Score: [Provide F1 score]
💻 How to Use

## Install Transformers Library:
```bash
pip install transformers
```
### Load Language-Specific Model:
``python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "cx-olquinjica/kimbundu-news-category-classification"  # Replace with the desired language model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```
### Classify News Topics:
```python
news_text = "Input your news text here."
inputs = tokenizer(news_text, return_tensors="pt")
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits).item()
```
### Explore Multilingual Capabilities:
Experiment with different languages and their respective models for accurate news topic classification.
🙌 Acknowledgments

The News-Topic-Classification models are built on the AngoBERTa architecture and were fine-tuned using the SIB-200 dataset.

## 🐞 Issues and Contributions

If you encounter any issues or have suggestions for improvements, please open an issue on the respective model's Hugging Face repository. Contributions are welcome!

## 📄 License

The models in this repository are licensed under the Apache 2.0 license.

Feel free to explore the News-Topic-Classification models for accurate and language-specific news categorization! 📰✨
