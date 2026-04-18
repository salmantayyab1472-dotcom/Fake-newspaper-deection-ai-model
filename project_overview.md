# Fake Newspaper Detection Model – Project Overview

## 1. Purpose
The **Fake Newspaper Detection** project aims to build a machine‑learning system that can automatically identify whether a given newspaper article (or a scanned image of a newspaper page) is genuine or fabricated.  This can be useful for:
- **Media verification** – flagging potentially manipulated news content.
- **Academic research** – studying the spread of misinformation.
- **Automation pipelines** – integrating with news aggregators to filter out fake articles.

## 2. High‑Level Architecture
```
Fake Newspaper Detection Model/
├─ data/                     # Raw & processed datasets
│   ├─ raw/                 # Original PDFs, images, CSVs
│   └─ processed/           # Tokenized text, extracted features
├─ notebooks/                # Exploratory analysis & experiments
├─ src/                      # Source code (Python package)
│   ├─ __init__.py
│   ├─ data/                # Data loading & preprocessing utilities
│   ├─ models/              # Model definitions (CNN, Transformer, etc.)
│   ├─ train.py             # Training script (CLI entry point)
│   ├─ evaluate.py          # Evaluation & metrics
│   └─ inference.py         # Inference API (REST or CLI)
├─ tests/                    # Unit / integration tests
├─ requirements.txt          # Python dependencies
├─ setup.cfg / pyproject.toml # Packaging configuration
├─ README.md                 # Project documentation (this file)
└─ .gitignore               # Ignored files
```

## 3. Data Sources & Preparation
| Source | Type | Description |
|--------|------|-------------|
| **Kaggle – Fake News Dataset** | CSV (title, text, label) | Text‑only fake/real news articles.
| **Newspaper3k Scraper** | HTML → Text | Real newspaper articles scraped from reputable sites.
| **Synthetic Images** | PNG/JPEG | Scanned newspaper page images (for OCR‑based pipeline).
| **Fact‑checking APIs** (e.g., Google Fact Check) | JSON | Additional label verification.

### 3.1 Text‑only Pipeline
1. **Collect** CSV files with `title`, `text`, `label` (0 = real, 1 = fake).
2. **Clean** – remove HTML tags, normalize whitespace, lower‑case.
3. **Tokenize** using `transformers` tokenizer (e.g., `bert-base-uncased`).
4. **Split** into train/validation/test (80/10/10).

### 3.2 Image‑based Pipeline (optional)
1. **Scan** newspaper pages → high‑resolution images.
2. **Run OCR** with `pytesseract` to extract text.
3. **Combine** extracted text with layout features (image size, font density).
4. **Feed** both image and text into a multimodal model (e.g., CLIP‑style).

## 4. Model Choices
| Approach | Library | Pros | Cons |
|----------|---------|------|------|
| **Logistic Regression / SVM** on TF‑IDF | scikit‑learn | Fast, interpretable | Limited performance on nuanced language.
| **BERT / RoBERTa fine‑tuning** | transformers | State‑of‑the‑art NLP, captures context | Requires GPU for reasonable training time.
| **CNN on Images + Text Fusion** | PyTorch / torchvision | Handles visual cues (e.g., altered layouts) | More complex, needs large image dataset.
| **Ensemble (BERT + Gradient Boosting)** | sklearn + transformers | Improves robustness | Higher inference latency.

### Example: BERT Fine‑tuning (`src/models/bert_classifier.py`)
```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class BertFakeNewsClassifier:
    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def predict(self, texts: list[str]):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        return probs
```

## 5. Training Script (`src/train.py`)
```python
import argparse
import torch
from torch.utils.data import DataLoader
from src.data.dataset import FakeNewsDataset
from src.models.bert_classifier import BertFakeNewsClassifier
from transformers import AdamW, get_linear_schedule_with_warmup

def main(args):
    dataset = FakeNewsDataset(args.train_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BertFakeNewsClassifier().model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(loader) * args.epochs
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{args.epochs} – loss: {loss.item():.4f}")
    torch.save(model.state_dict(), args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--output_path", default="model.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
```

## 6. Evaluation (`src/evaluate.py`)
- **Metrics**: Accuracy, Precision, Recall, F1‑score, ROC‑AUC.
- **Confusion Matrix** visualisation with `seaborn`.
- **Cross‑validation** (5‑fold) for robust performance estimation.

## 7. Inference API (`src/inference.py`)
A lightweight Flask endpoint to serve predictions:
```python
from flask import Flask, request, jsonify
from src.models.bert_classifier import BertFakeNewsClassifier

app = Flask(__name__)
model = BertFakeNewsClassifier()
model.model.eval()

@app.post("/predict")
def predict():
    data = request.get_json()
    texts = data.get("texts", [])
    probs = model.predict(texts)
    return jsonify({"probabilities": probs.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
```

## 8. Development Workflow
1. **Create a virtual environment** inside the project folder:
   ```powershell
   cd "C:\Users\shera\Desktop\Fake newspaper detecion model"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. **Run notebooks** for EDA (`notebooks/eda.ipynb`).
3. **Train** the model:
   ```powershell
   python src/train.py --train_csv data/processed/train.csv --output_path models/bert_fake_news.pt
   ```
4. **Evaluate**:
   ```powershell
   python src/evaluate.py --model_path models/bert_fake_news.pt --test_csv data/processed/test.csv
   ```
5. **Serve** predictions:
   ```powershell
   python src/inference.py
   ```

## 9. Testing & CI
- **Unit tests** (`tests/`) covering data loading, model forward pass, and API response.
- **GitHub Actions** workflow to run tests on each push and optionally build a Docker image.

## 10. Future Enhancements
- **Multimodal fusion** of image + text (using CLIP or ViLT).
- **Explainability** – integrate `captum` or `LIME` to highlight words that contributed to a fake prediction.
- **Continuous learning** – periodic retraining with newly labeled articles.
- **Deployment** – containerize the Flask inference service with Docker and expose via a Kubernetes service.

---
*This document provides a complete conceptual and practical overview of the Fake Newspaper Detection project, ready to be fleshed out with actual data and code.*
