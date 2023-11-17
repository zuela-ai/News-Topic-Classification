# News-Topic-Classification Model Card

## 📰 Introduction

Welcome to the News-Topic-Classification repository, a collection of language-specific models for classifying news topics. Each model is fine-tuned for a specific language, including Kimbundu, Umbundu, Kikongo, Chokwe, and Lua. These models are derived from the AngoBERTa architecture and were trained on the SIB-200 dataset, which can be found here.

## 🌍 Language Models

- **Kimbundu**:
  - **Model**: [Hugging Face](https://huggingface.co/cx-olquinjica/kimbundu-news-category-classification)
- **Umbundu**:
  - **Model**: [Hugging Face](https://huggingface.co/cx-olquinjica/umbundu-news-category-classification) 
- **Kikongo**:
  - **Model** : [Hugging Face](https://huggingface.co/cx-olquinjica/kikongo-news-category-classification)
- **Chokwe**:
  - **Model**: [Hugging Face](https://huggingface.co/cx-olquinjica/chokwe-news-category-classification)
- **Lua**:
  - **Model**: [Hugging Face](https://huggingface.co/cx-olquinjica/lua-news-category-classification)
## 🚀 Training Details

- **Base Model**: [AngoBERTa](https://huggingface.co/cx-olquinjica/AngoBERTa)
- **Dataset**: [SIB-200 Dataset](https://github.com/dadelani/sib-200)
## 📊 Evaluation Metrics (Sample)

- Epochs: 3.0
- Training Loss: [Provide training loss]
- Evaluation Loss: [Provide evaluation loss]
- Accuracy: [Provide accuracy]
- Precision: [Provide precision]
- Recall: [Provide recall]
- F1 Score: [Provide F1 score]
  ## 💻 How to Use

## Install Transformers Library:
```bash
pip install transformers
```
### Load Language-Specific Model:
```python
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
