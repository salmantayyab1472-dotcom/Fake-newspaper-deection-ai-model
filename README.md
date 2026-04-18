# Fake-newspaper-deection-ai-model

A Flask web application that detects whether a news article is real or fake using a MobileNetV2 + Bi‑LSTM pipeline.

## Quick start

## 📂 Project Structure

Fake newspaper detecion model/
│
├─ .git/                     # Git metadata (auto‑created by `git init`)
├─ .gitignore                # already in repo
├─ README.md                 # already in repo
├─ requirements.txt          # already in repo
├─ run_example.py            # already in repo
│
├─ app.py                    # Flask entry point (already edited)
│
├─ src/                      # Python package with your model / utils
│   ├─ __init__.py           # makes `src` a package (already exists)
│   ├─ inference.py          # your model inference code
│   ├─ train.py              # training script
│   └─ ... (any other .py)   # keep all model‑related code here
│
├─ templates/                # HTML templates
│   ├─ index.html            # main page (already there)
│   └─ static/               # static assets (CSS, JS, images)
│       ├─ style.css         # dark‑mode glassmorphism CSS (created)
│       └─ (optional) script.js, logo.png, etc.
│
└─ data/ (optional)         # if you ever want to store small CSVs / JSONs
    └─ raw/                  # keep large raw data out of Git (via .gitignore)


```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Features
- Glass‑morphism UI with dark mode
- Confidence bar with animated fill
- Flash messages for upload status

## License
MIT
