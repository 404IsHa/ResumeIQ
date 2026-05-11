import streamlit as st
import PyPDF2
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ · AI Resume Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root & Reset ── */
:root {
  --bg:        #0a0a0f;
  --surface:   #111118;
  --card:      #16161f;
  --border:    #2a2a3a;
  --accent:    #7c6cfc;
  --accent2:   #fc6c8f;
  --accent3:   #6cfcd8;
  --text:      #e8e8f0;
  --muted:     #6b6b80;
  --mono:      'Space Mono', monospace;
  --sans:      'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(124,108,252,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(252,108,143,0.08) 0%, transparent 60%),
    var(--bg);
}

/* hide Streamlit chrome */
#MainMenu, footer, header, [data-testid="stToolbar"] { display:none !important; }
[data-testid="stDecoration"] { display:none !important; }

/* ── Typography ── */
h1,h2,h3,h4 { font-family: var(--mono) !important; }
p, li, div  { font-family: var(--sans) !important; }

/* ── Hero ── */
.hero {
  text-align: center;
  padding: 3rem 0 2rem;
}
.hero-badge {
  display: inline-block;
  background: rgba(124,108,252,0.15);
  border: 1px solid rgba(124,108,252,0.4);
  color: #a89ffd;
  font-family: var(--mono);
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  padding: 5px 14px;
  border-radius: 100px;
  margin-bottom: 1.2rem;
}
.hero h1 {
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 700;
  margin: 0 0 0.6rem;
  background: linear-gradient(135deg, #fff 0%, #a89ffd 50%, #fc6c8f 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1.1;
}
.hero p {
  color: var(--muted);
  font-size: 1.05rem;
  font-weight: 300;
  max-width: 520px;
  margin: 0 auto;
}

/* ── Cards ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.6rem;
  margin-bottom: 1.2rem;
  position: relative;
  overflow: hidden;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(124,108,252,0.5), transparent);
}
.card-label {
  font-family: var(--mono);
  font-size: 0.68rem;
  letter-spacing: 0.15em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 0.8rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Score Gauge ── */
.score-wrap {
  text-align: center;
  padding: 2rem 0 1rem;
}
.score-ring {
  position: relative;
  width: 160px;
  height: 160px;
  margin: 0 auto 1rem;
}
.score-ring svg { transform: rotate(-90deg); }
.score-ring .track { stroke: var(--border); }
.score-ring .fill  { stroke-linecap: round; transition: stroke-dashoffset 1s ease; }
.score-number {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.score-number span:first-child {
  font-family: var(--mono);
  font-size: 2.4rem;
  font-weight: 700;
  line-height: 1;
}
.score-number span:last-child {
  font-size: 0.75rem;
  color: var(--muted);
  font-family: var(--mono);
  letter-spacing: 0.1em;
}
.score-label {
  font-family: var(--mono);
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 0.5rem;
}

/* ── Keyword Pills ── */
.pill-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 0.8rem;
}
.pill {
  font-family: var(--mono);
  font-size: 0.72rem;
  padding: 4px 12px;
  border-radius: 100px;
  border: 1px solid;
}
.pill-miss  { color: #fc6c8f; border-color: rgba(252,108,143,0.35); background: rgba(252,108,143,0.07); }
.pill-match { color: #6cfcd8; border-color: rgba(108,252,216,0.35); background: rgba(108,252,216,0.07); }

/* ── Suggestions ── */
.suggestion {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  padding: 0.85rem 0;
  border-bottom: 1px solid var(--border);
}
.suggestion:last-child { border-bottom: none; }
.sug-icon {
  width: 30px; height: 30px;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem;
  flex-shrink: 0;
  margin-top: 2px;
}
.sug-title { font-weight: 600; font-size: 0.88rem; margin-bottom: 2px; }
.sug-body  { font-size: 0.82rem; color: var(--muted); line-height: 1.5; }

/* ── Stat boxes ── */
.stat-row {
  display: flex;
  gap: 12px;
  margin-top: 1rem;
}
.stat-box {
  flex: 1;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  text-align: center;
}
.stat-val { font-family: var(--mono); font-size: 1.5rem; font-weight: 700; }
.stat-lbl { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }

/* ── Upload zone overrides ── */
[data-testid="stFileUploader"] {
  border: 1px dashed var(--border) !important;
  border-radius: 12px !important;
  padding: 0.5rem !important;
  background: rgba(255,255,255,0.02) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent) !important;
}

/* ── Textarea overrides ── */
textarea {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
  font-family: var(--sans) !important;
  font-size: 0.88rem !important;
}
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(124,108,252,0.15) !important; }

/* ── Button ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent), #9c6cfc) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: var(--mono) !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  padding: 0.7rem 2rem !important;
  width: 100% !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 20px rgba(124,108,252,0.35) !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 30px rgba(124,108,252,0.5) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── Progress bar ── */
.section-bar {
  height: 4px;
  border-radius: 4px;
  background: var(--border);
  margin: 6px 0 12px;
  overflow: hidden;
}
.section-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 1s ease;
}

/* ── Footer ── */
.footer {
  text-align: center;
  padding: 2rem 0 1rem;
  color: var(--muted);
  font-size: 0.78rem;
  font-family: var(--mono);
}
</style>
""", unsafe_allow_html=True)

# ── NLP helpers ────────────────────────────────────────────────────────────────
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'as','by','from','is','are','was','were','be','been','being','have',
    'has','had','do','does','did','will','would','could','should','may',
    'might','shall','can','need','dare','ought','used','it','its','this',
    'that','these','those','i','you','he','she','we','they','me','him',
    'her','us','them','my','your','his','our','their','what','which','who',
    'when','where','why','how','all','each','every','both','few','more',
    'most','other','some','such','no','not','only','same','so','than',
    'too','very','s','t','just','about','up','also','into','through','during',
    'before','after','above','below','between','own','off','over','under',
    'again','further','then','once'
}

TECH_SKILLS = {
    # Languages
    'python','java','javascript','typescript','c++','c#','go','rust','kotlin',
    'swift','ruby','php','scala','r','matlab','sql','html','css','bash','shell',
    # Frameworks/Libraries
    'react','angular','vue','django','flask','fastapi','spring','nodejs','express',
    'tensorflow','pytorch','keras','pandas','numpy','scikit-learn','sklearn',
    'streamlit','nextjs','nestjs','graphql','rest','restful',
    # Cloud & DevOps
    'aws','azure','gcp','docker','kubernetes','terraform','ci/cd','jenkins',
    'github','gitlab','git','linux','unix','nginx','apache',
    # Data
    'machine learning','deep learning','nlp','computer vision','data science',
    'postgresql','mysql','mongodb','redis','elasticsearch','kafka','spark',
    'hadoop','tableau','powerbi','excel',
    # Concepts
    'agile','scrum','microservices','api','oop','solid','tdd','devops',
    'blockchain','cybersecurity','ml','ai','llm','rag',
}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s/#+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str):
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]

def extract_tech_keywords(text: str) -> set:
    text_lower = text.lower()
    found = set()
    # multi-word first
    for skill in TECH_SKILLS:
        if skill in text_lower:
            found.add(skill)
    return found

def compute_similarity(resume: str, jd: str) -> float:
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    try:
        tfidf = vec.fit_transform([resume, jd])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return round(float(score) * 100, 1)
    except Exception:
        return 0.0

def get_important_jd_keywords(jd_clean: str, top_n: int = 30) -> list:
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=200)
    try:
        vec.fit_transform([jd_clean])
        feature_names = vec.get_feature_names_out()
        scores = vec.idf_
        pairs = sorted(zip(feature_names, scores), key=lambda x: -x[1])
        return [w for w, _ in pairs[:top_n] if len(w) > 2]
    except Exception:
        return []

def score_color(score: float):
    if score >= 75: return "#6cfcd8", "Excellent Match 🚀"
    if score >= 55: return "#f9d56e", "Good Match ✨"
    if score >= 35: return "#fc8f6c", "Fair Match 🔧"
    return "#fc6c8f", "Needs Work 💪"

def generate_suggestions(missing: list, score: float, resume_words: set, jd_words: set) -> list:
    suggestions = []

    if missing:
        top = missing[:6]
        suggestions.append({
            "icon": "🎯", "color": "rgba(252,108,143,0.15)", "border": "rgba(252,108,143,0.3)",
            "title": "Add Missing Technical Skills",
            "body": f"Incorporate these high-impact keywords naturally: {', '.join(top)}. Add them to your Skills section or weave them into bullet points."
        })

    if score < 55:
        suggestions.append({
            "icon": "📝", "color": "rgba(249,213,110,0.12)", "border": "rgba(249,213,110,0.3)",
            "title": "Mirror the Job Description Language",
            "body": "Use the exact phrasing from the JD. ATS systems do exact keyword matching — 'led' vs 'managed' can make a difference."
        })

    suggestions.append({
        "icon": "📊", "color": "rgba(108,252,216,0.1)", "border": "rgba(108,252,216,0.25)",
        "title": "Quantify Your Achievements",
        "body": "Add numbers, percentages, and impact metrics. E.g., 'Improved model accuracy by 18%' beats 'Improved model accuracy'."
    })

    if score >= 55:
        suggestions.append({
            "icon": "✏️", "color": "rgba(124,108,252,0.12)", "border": "rgba(124,108,252,0.3)",
            "title": "Write a Tailored Summary",
            "body": "Add a 2-3 line professional summary at the top that directly echoes the role's core requirements. This boosts ATS ranking significantly."
        })

    suggestions.append({
        "icon": "🔧", "color": "rgba(252,143,108,0.1)", "border": "rgba(252,143,108,0.25)",
        "title": "Format for ATS Compatibility",
        "body": "Use standard section headings (Experience, Education, Skills). Avoid tables, columns, or graphics — ATS parsers can't read them."
    })

    return suggestions

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">⚡ AI POWERED · NLP ENGINE</div>
  <h1>ResumeIQ</h1>
  <p>Instantly score your resume against any job description and get actionable feedback to land more interviews.</p>
</div>
""", unsafe_allow_html=True)

# ── Two-column input layout ────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="card-label">📄 YOUR RESUME</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF Resume", type=["pdf"], label_visibility="collapsed")
    if uploaded:
        st.markdown(f'<p style="font-size:0.8rem;color:#6cfcd8;font-family:var(--mono)">✓ {uploaded.name} loaded</p>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card-label">💼 JOB DESCRIPTION</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Paste the job description here",
        height=220,
        placeholder="Paste the full job description here…\n\nInclude requirements, responsibilities, and preferred qualifications for the best analysis.",
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze = st.button("⚡  ANALYZE MY RESUME", use_container_width=True)

# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze:
    if not uploaded:
        st.error("Please upload your resume PDF.")
    elif not jd_text.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Running NLP analysis…"):
            # Extract
            raw_resume = extract_text_from_pdf(uploaded.read())
            raw_jd = jd_text

            clean_resume = clean_text(raw_resume)
            clean_jd    = clean_text(raw_jd)

            res_words = set(tokenize(clean_resume))
            jd_words  = set(tokenize(clean_jd))

            # Score
            score = compute_similarity(clean_resume, clean_jd)

            # Keyword analysis
            res_tech  = extract_tech_keywords(raw_resume)
            jd_tech   = extract_tech_keywords(raw_jd)
            matched   = res_tech & jd_tech
            missing   = jd_tech - res_tech

            # Important JD keywords (non-tech)
            jd_important = get_important_jd_keywords(clean_jd, top_n=40)
            missing_general = [k for k in jd_important if k not in clean_resume and k not in [m.lower() for m in missing]][:12]

            total_missing = list(missing) + missing_general[:max(0, 12 - len(missing))]

        # ── Results ────────────────────────────────────────────────────────────
        st.markdown("---")
        accent_color, score_label = score_color(score)
        circumference = 2 * 3.14159 * 62
        dash_offset   = circumference * (1 - score / 100)

        r1, r2 = st.columns([1, 2], gap="large")

        with r1:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
              <div class="card-label" style="justify-content:center;">MATCH SCORE</div>
              <div class="score-ring">
                <svg viewBox="0 0 140 140" width="160" height="160">
                  <circle class="track" cx="70" cy="70" r="62"
                    fill="none" stroke-width="10"/>
                  <circle class="fill" cx="70" cy="70" r="62"
                    fill="none" stroke="{accent_color}" stroke-width="10"
                    stroke-dasharray="{circumference:.1f}"
                    stroke-dashoffset="{dash_offset:.1f}"/>
                </svg>
                <div class="score-number">
                  <span style="color:{accent_color}">{score}%</span>
                  <span>MATCH</span>
                </div>
              </div>
              <div class="score-label" style="color:{accent_color}">{score_label}</div>
              <div class="stat-row">
                <div class="stat-box">
                  <div class="stat-val" style="color:#6cfcd8">{len(matched)}</div>
                  <div class="stat-lbl">Skills Matched</div>
                </div>
                <div class="stat-box">
                  <div class="stat-val" style="color:#fc6c8f">{len(missing)}</div>
                  <div class="stat-lbl">Skills Missing</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            # Matched keywords
            if matched:
                st.markdown('<div class="card-label">✅ MATCHED KEYWORDS</div>', unsafe_allow_html=True)
                pills = "".join(f'<span class="pill pill-match">{k}</span>' for k in sorted(matched))
                st.markdown(f'<div class="pill-grid">{pills}</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # Missing keywords
            all_missing_display = list(missing)[:18]
            if all_missing_display:
                st.markdown('<div class="card-label">❌ MISSING KEYWORDS</div>', unsafe_allow_html=True)
                pills = "".join(f'<span class="pill pill-miss">{k}</span>' for k in sorted(all_missing_display))
                st.markdown(f'<div class="pill-grid">{pills}</div>', unsafe_allow_html=True)

        # ── Section coverage bars ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">📊 SECTION ANALYSIS</div>', unsafe_allow_html=True)

        sections = {
            "Skills":        ("skills" in clean_resume or "technologies" in clean_resume, "#7c6cfc"),
            "Experience":    ("experience" in clean_resume or "work" in clean_resume, "#6cfcd8"),
            "Education":     ("education" in clean_resume or "university" in clean_resume or "degree" in clean_resume, "#f9d56e"),
            "Certifications":("certif" in clean_resume or "aws" in clean_resume or "google" in clean_resume, "#fc8f6c"),
            "Projects":      ("project" in clean_resume, "#fc6c8f"),
        }
        cols = st.columns(5)
        for i, (section, (found, color)) in enumerate(sections.items()):
            with cols[i]:
                status = "✓ Found" if found else "✗ Missing"
                c = color if found else "#6b6b80"
                st.markdown(f"""
                <div style="text-align:center;padding:0.7rem;background:rgba(255,255,255,0.03);border:1px solid {'rgba(108,252,216,0.2)' if found else 'var(--border)'};border-radius:10px;">
                  <div style="font-family:var(--mono);font-size:1.3rem;margin-bottom:4px">{'✓' if found else '✗'}</div>
                  <div style="font-weight:600;font-size:0.82rem;color:{c}">{section}</div>
                  <div style="font-size:0.7rem;color:var(--muted);margin-top:2px">{status}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Suggestions ────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-label">💡 IMPROVEMENT SUGGESTIONS</div>', unsafe_allow_html=True)

        suggestions = generate_suggestions(list(missing), score, res_words, jd_words)
        for s in suggestions:
            st.markdown(f"""
            <div class="suggestion">
              <div class="sug-icon" style="background:{s['color']};border:1px solid {s['border']}">{s['icon']}</div>
              <div>
                <div class="sug-title">{s['title']}</div>
                <div class="sug-body">{s['body']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Quick tips ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2, t3 = st.columns(3)
        tips = [
            ("🎯", "#7c6cfc", "One resume per job", "Never send a generic resume. Tailor every application to the specific role."),
            ("🤖", "#6cfcd8", "Beat the ATS first", "70% of resumes are rejected before a human sees them. Keywords are your first gate."),
            ("📈", "#f9d56e", "Target 60–80% score", "A perfect 100% may look keyword-stuffed. Aim for natural, high-relevance language."),
        ]
        for col, (icon, color, title, body) in zip([t1, t2, t3], tips):
            with col:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:12px;padding:1.1rem;height:100%">
                  <div style="font-size:1.5rem;margin-bottom:0.5rem">{icon}</div>
                  <div style="font-weight:600;font-size:0.88rem;color:{color};margin-bottom:0.3rem">{title}</div>
                  <div style="font-size:0.8rem;color:var(--muted);line-height:1.55">{body}</div>
                </div>
                """, unsafe_allow_html=True)

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:var(--muted)">
      <div style="font-size:3rem;margin-bottom:1rem">📄</div>
      <div style="font-family:var(--mono);font-size:0.85rem;letter-spacing:0.1em">
        UPLOAD YOUR RESUME & PASTE A JOB DESCRIPTION TO BEGIN
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="footer">ResumeIQ · Built with Python · NLP · TF-IDF · Cosine Similarity</div>', unsafe_allow_html=True)
