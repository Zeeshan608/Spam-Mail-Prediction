# 📧 Spam Mail Prediction

A machine learning web app that classifies email/SMS messages as **Spam** or **Ham**, using TF-IDF text vectorization and a Multinomial Naive Bayes classifier. Built with **scikit-learn** for modeling and **Streamlit** for a clean, interactive front end.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

Spam detection is one of the most widely used applications of natural language processing in everyday software — every major email provider runs some version of it behind the scenes. This project trains a **Multinomial Naive Bayes** classifier on TF-IDF features extracted from a labeled dataset of SMS/email messages, then wraps it in a Streamlit app so anyone can paste in a message and instantly see whether it's spam, complete with a confidence score and a breakdown of which words influenced the decision.

> ⚠️ **Disclaimer:** This is a demo classifier trained on a public dataset. It's a great showcase of an NLP classification pipeline, but it isn't a substitute for the spam filtering built into real email providers.

---

## ✨ Features

- 🎯 **Instant classification** — Spam / Ham with a confidence percentage
- 📊 **Visual spam-likelihood gauge** for at-a-glance interpretation
- 🔍 **Explainability panel** — see which specific words in the message pushed it toward spam or ham
- 🖥️ **Clean, card-based UI** with a modern gradient design
- ⚡ **One-click examples** to try a sample spam or ham message
- 🧠 **TF-IDF + Naive Bayes pipeline** — a classic, fast, and highly interpretable NLP approach

---

## 🖼️ Demo

<p align="center">
  <img src="assets/screenshot.png" alt="App screenshot" width="800">
</p>

> Replace `<img width="960" height="475" alt="image" src="https://github.com/user-attachments/assets/d8b1e4ce-fbb1-4a3a-b604-cb48c517a22a" />
` with an actual screenshot of your running app.

---

## 🧮 Model Details

| Item | Description |
|---|---|
| **Algorithm** | Multinomial Naive Bayes (scikit-learn) |
| **Feature Extraction** | TF-IDF (`TfidfVectorizer`, English stop words removed) |
| **Target** | `Category` (Spam / Ham) |
| **Train/Test Split** | 80% / 20% |
| **Vocabulary Size** | 8,440 features |
| **Dataset Size** | 5,572 messages |

### Performance on the held-out test set

| Metric | Score |
|---|---|
| Accuracy | 97.76% |
| Spam Precision | 100% |
| Spam Recall | 83% |
| Spam F1-Score | 91% |
| Ham Precision | 97% |
| Ham Recall | 100% |

---

## 🗂️ Project Structure

```
Spam-Mail-Prediction/
├── app.py                              # Streamlit front-end
├── Spam_Mail_Detection.ipynb           # Data cleaning, EDA, training & evaluation
├── spam_ham_model.pkl                  # Trained model (pickle format)
├── spam_ham_model.joblib               # Trained model (joblib format)
├── tfidf_vectorizer.pkl                # Fitted TF-IDF vectorizer
├── mail_data.csv                       # Training dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or later
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Zeeshan608/Spam-Mail-Prediction.git
cd Spam-Mail-Prediction

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

---

## 🧪 Retraining the Model

The full data cleaning, TF-IDF vectorization, training, and evaluation pipeline lives in `Spam_Mail_Detection.ipynb`. Run it end-to-end to regenerate `spam_ham_model.pkl`, `spam_ham_model.joblib`, and `tfidf_vectorizer.pkl` if you update the dataset or want to try a different algorithm (e.g. Logistic Regression or SVM, both already imported in the notebook).

---

## 🛠️ Tech Stack

- **Python** — core language
- **pandas / NumPy** — data manipulation
- **scikit-learn** — TF-IDF vectorization, model training and evaluation
- **Streamlit** — interactive web front end
- **Plotly** — spam-likelihood gauge visualization

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Zeeshan Ahmad Akhtar**
[GitHub](https://github.com/Zeeshan608)

If you found this project useful, consider giving it a ⭐ on GitHub!
