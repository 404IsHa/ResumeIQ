# ⚡ ResumeIQ — AI Resume Analyzer

> Upload your resume → Paste a job description → Get an instant match score + actionable suggestions.

---

## 🚀 Quick Start

### 1. Clone / download this project
```bash
git clone <your-repo-url>
cd resume_analyzer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501` 🎉

---

## 🧠 How It Works

| Step | What happens |
|------|-------------|
| 1 | **PDF Extraction** — `PyPDF2` pulls raw text from your resume |
| 2 | **Text Cleaning** — lowercasing, punctuation removal, stopword filtering |
| 3 | **TF-IDF Vectorization** — both texts converted to numerical vectors (bigrams) |
| 4 | **Cosine Similarity** — measures directional closeness of the two vectors |
| 5 | **Keyword Gap Analysis** — 50+ tech skills checked against both documents |
| 6 | **Suggestions** — context-aware improvement tips generated from results |

---

## ✨ Features

- 📊 **Match Score (%)** — TF-IDF + Cosine Similarity score
- ✅ **Matched Keywords** — skills found in both resume & JD  
- ❌ **Missing Keywords** — high-value skills to add
- 🏗️ **Section Detector** — checks for Skills, Experience, Education, Projects, Certifications
- 💡 **Smart Suggestions** — tailored advice based on your specific gap
- 🎨 **Polished dark UI** — professional enough to screenshot for your portfolio

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| NLP / ML | scikit-learn (TF-IDF, Cosine Similarity) |
| Keyword Matching | Custom skill dictionary (50+ tech terms) |
| Styling | Custom CSS (dark theme, Google Fonts) |

---

## 📂 Project Structure

```
resume_analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 💡 Tips for Best Results

- Use a **text-based PDF** (not a scanned image)
- Paste the **full job description** including requirements & responsibilities
- Aim for a **60–80% match score** — 100% looks keyword-stuffed
- **Tailor your resume** for each application using the missing keywords

---

## 🌟 Resume Talking Points

When explaining this project in interviews:
- *"Built an NLP pipeline using TF-IDF vectorization and cosine similarity to quantify semantic overlap between resume and job description text."*
- *"Implemented a keyword gap analysis engine that identifies missing technical skills against a curated domain-specific lexicon."*
- *"Designed a responsive, production-grade Streamlit interface with custom CSS theming."*

---

*Built with Python · Streamlit · scikit-learn · PyPDF2*
