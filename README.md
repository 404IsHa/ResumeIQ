# ⚡ ResumeIQ — AI Resume Analyzer

> Upload your resume → Paste a job description → Get a match score, a skill gap and practical suggestions.

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

Your browser will open at `http://localhost:8501`.

---

## 🧠 How It Works

| Step | What happens |
|------|-------------|
| 1 | **PDF extraction**: `pypdf` reads the text of a text-based PDF resume |
| 2 | **Text cleaning**: lowercasing and punctuation removal (symbols like `+`, `#`, `/` are kept for C++, C#, CI/CD) |
| 3 | **TF-IDF vectorization**: resume and job description become numeric vectors (unigrams + bigrams). The vocabulary comes from the job description only, so resume-only words do not dilute the score |
| 4 | **Cosine similarity**: measures how close the two vectors are, shown as the match score |
| 5 | **Keyword gap analysis**: about 90 technical skills are searched with whole-word matching, including aliases (e.g. ML = machine learning) |
| 6 | **Section check**: looks for headings such as Skills, Experience, Education, Projects, Certifications |
| 7 | **Suggestions**: rule-based tips based on the score, the missing skills and the missing sections |

The match score measures **keyword (lexical) overlap**, not meaning. For example, it cannot tell that two different words mean the same thing unless they are in the alias list.

---

## ✨ Features

- 📊 **Match score (%)**: TF-IDF + cosine similarity
- ✅ **Matched keywords** and ❌ **missing keywords**
- 📈 **Skill coverage**: share of the JD's skills that your resume has
- 🏗️ **Section analysis**: heading-based check
- 💡 **Rule-based suggestions**
- 🎨 **Dark custom UI** built with Streamlit and custom CSS

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit + custom CSS |
| PDF parsing | pypdf |
| NLP / ML | scikit-learn (TF-IDF, cosine similarity) |
| Keyword matching | Skill dictionary with aliases and regex whole-word matching |

---

## 📂 Project Structure

```
resume_analyzer/
├── app.py              # Streamlit application (UI + NLP logic)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ⚠️ Limitations

- Works with **text-based PDFs** only (not scanned images)
- Keyword based: it does not understand meaning or the quality of your experience
- The skill list is manual, so skills outside the list are not detected
- The score cut-offs (Excellent / Good / Fair) are heuristic, not calibrated on labelled data
- Results disappear if you interact with the page after analysing (no session state yet)

## 🔭 Future Improvements

- Sentence embeddings for semantic matching
- Automatic skill extraction instead of a fixed list
- OCR support for scanned PDFs
- Evaluation on labelled resume–job pairs
- Keep results with `st.session_state`

---

*Built with Python · Streamlit · scikit-learn · pypdf*
