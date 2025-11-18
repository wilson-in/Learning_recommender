# app.py — Learning Recommender (OpenRouter-only, patched)
# - Remote embeddings: OpenRouter (preferred) via OPENROUTER_API_KEY
# - Fallbacks: local sentence-transformers -> TF-IDF
# - Robust CSV loader (tolerant to unquoted commas)
# - Realistic fit normalization (no flat 90s)
# - Resume parser shows email but hides phone in UI
# - Career roadmap & visualization included
#
# Usage:
#   - Put this app.py and courses_extended.csv in the same folder
#   - Optionally set OPENROUTER_API_KEY in your environment to enable OpenRouter embeddings
#   - Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import traceback
from datetime import datetime, timedelta
from math import ceil
import time
import requests
import csv
from io import StringIO

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="Learning Recommender", initial_sidebar_state="expanded")

# ---------------------------
# Robust CSV loader
# ---------------------------
@st.cache_data
def load_catalog(path='courses_extended.csv'):
    """
    Robust CSV loader:
    - Try pandas.read_csv with python engine and on_bad_lines='warn'
    - If it fails, parse with csv.reader and repair rows so each row has the same number of columns as header.
    """
    try:
        df = pd.read_csv(path, engine='python', on_bad_lines='warn', skipinitialspace=True)
    except Exception as e:
        df = None
        print("Initial pandas.read_csv failed:", repr(e))

    if isinstance(df, pd.DataFrame) and len(df.columns) >= 5:
        # post-processing
        df['prerequisites'] = df.get('prerequisites', 'none').fillna('none')
        df['skill_tags'] = df.get('skill_tags', '').fillna('')
        df['outcomes'] = df.get('outcomes', '').fillna('')
        df['level'] = df.get('level', 'beginner').fillna('beginner')
        df['popularity_score'] = df.get('popularity_score', 5).fillna(5)
        df['is_free'] = df.get('is_free', 'yes').fillna('yes')
        df['free_link'] = df.get('free_link', df['link'] if 'link' in df.columns else '')
        df['first_project'] = df.get('first_project', pd.Series([""] * len(df)))
        df['hours_per_week'] = df.get('hours_per_week', 8).fillna(8)
        df['cost_estimate_usd'] = df.get('cost_estimate_usd', 0).fillna(0)
        df['program_type'] = df.get('program_type', 'certification').fillna('certification')
        df['mode'] = df.get('mode', 'self-paced').fillna('self-paced')
        df['credential'] = df.get('credential', 'certificate').fillna('certificate')
        if 'id' not in df.columns:
            df.insert(0, 'id', ['c'+str(i+1) for i in range(len(df))])
        return df

    # fallback manual parse and repair
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    sio = StringIO(raw)
    reader = csv.reader(sio)
    rows = list(reader)
    if not rows:
        return pd.DataFrame()

    header = rows[0]
    ncols = len(header)
    repaired = [header]
    for row in rows[1:]:
        if len(row) == ncols:
            repaired.append(row)
        elif len(row) > ncols:
            new_row = row[:ncols-1] + [",".join(row[ncols-1:])]
            repaired.append(new_row)
        else:
            new_row = row + [''] * (ncols - len(row))
            repaired.append(new_row)

    df = pd.DataFrame(repaired[1:], columns=repaired[0])

    # post-processing
    df['prerequisites'] = df.get('prerequisites', 'none').fillna('none')
    df['skill_tags'] = df.get('skill_tags', '').fillna('')
    df['outcomes'] = df.get('outcomes', '').fillna('')
    df['level'] = df.get('level', 'beginner').fillna('beginner')
    df['popularity_score'] = df.get('popularity_score', 5).fillna(5)
    df['is_free'] = df.get('is_free', 'yes').fillna('yes')
    df['free_link'] = df.get('free_link', df['link'] if 'link' in df.columns else '')
    df['first_project'] = df.get('first_project', pd.Series([""] * len(df)))
    df['hours_per_week'] = df.get('hours_per_week', 8).fillna(8)
    df['cost_estimate_usd'] = df.get('cost_estimate_usd', 0).fillna(0)
    df['program_type'] = df.get('program_type', 'certification').fillna('certification')
    df['mode'] = df.get('mode', 'self-paced').fillna('self-paced')
    df['credential'] = df.get('credential', 'certificate').fillna('certificate')
    if 'id' not in df.columns:
        df.insert(0, 'id', ['c'+str(i+1) for i in range(len(df))])
    return df

# ---------------------------
# Helpers: scoring, timeline, rationales
# ---------------------------
def simple_prereq_penalty(user_skills: list, prereq_str: str):
    if not prereq_str or str(prereq_str).strip().lower() == 'none':
        return 0.0, []
    prereqs = [p.strip().lower() for p in str(prereq_str).replace(';', ',').split(',') if p.strip()]
    matched = 0
    missing = []
    for p in prereqs:
        found = False
        for s in user_skills:
            s_norm = s.split('(')[0].strip().lower()
            if s_norm and (s_norm in p or p in s_norm):
                matched += 1
                found = True
                break
        if not found:
            missing.append(p)
    total = len(prereqs)
    miss_count = max(0, total - matched)
    penalty = (miss_count / max(1, total)) * 6.0
    return float(penalty), missing

def level_bonus_score(user_level_guess: str, course_level: str):
    levels = {'none': 0, 'beginner': 1, 'beginner-intermediate': 1.5, 'intermediate': 2, 'advanced': 3}
    u = levels.get(user_level_guess.lower(), 1.5)
    c = levels.get(str(course_level).lower(), 1.5)
    if u >= c:
        return 10.0
    if u + 0.5 >= c:
        return 5.0
    return 0.0

def deterministic_rationale(course_row, profile_text, matched_skills):
    mapped = f"Maps {matched_skills or 'your skills'} → {course_row.get('outcomes','').split(',')[0] if course_row.get('outcomes') else course_row.get('title')}"
    gap = f"Fills: {course_row.get('outcomes','').split(',')[0] if course_row.get('outcomes') else 'skill gap'}; Prep: {course_row.get('prerequisites','none') if course_row.get('prerequisites','none')!='none' else 'none'}"
    return f"{mapped}. {gap}."

def infer_prereq_links(df):
    adj = {row['id']: set() for _, row in df.iterrows()}
    rows = df.to_dict(orient='records')
    for r in rows:
        cid = r['id']
        prereq_str = str(r.get('prerequisites', '')).lower()
        if not prereq_str or prereq_str.strip() == 'none':
            continue
        prereqs = [p.strip() for p in prereq_str.replace(';',',').split(',') if p.strip()]
        for p in prereqs:
            for other in rows:
                if other['id'] == cid:
                    continue
                search_text = (" ".join([str(other.get('skill_tags','')), str(other.get('title',''))])).lower()
                if p in search_text:
                    adj[other['id']].add(cid)
    return adj

def build_ordered_path(top_courses, df):
    adj = infer_prereq_links(df)
    top_ids = [c['id'] for c in top_courses]
    sub_adj = {tid: set() for tid in top_ids}
    indegree = {tid:0 for tid in top_ids}
    for u in top_ids:
        for v in adj.get(u, set()):
            if v in top_ids:
                sub_adj[u].add(v)
                indegree[v] += 1
    queue = [n for n,d in indegree.items() if d==0]
    ordered = []
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for neigh in list(sub_adj.get(node, [])):
            indegree[neigh] -= 1
            sub_adj[node].remove(neigh)
            if indegree[neigh] == 0:
                queue.append(neigh)
    remaining = [n for n in top_ids if n not in ordered]
    if remaining:
        ordered.extend(remaining)
    id_to_course = {c['id']: c for c in top_courses}
    ordered_courses = [id_to_course[i] for i in ordered if i in id_to_course]
    return ordered_courses

def estimate_weeks(course_row):
    try:
        dw = int(course_row.get('duration_weeks', 0))
    except Exception:
        dw = 0
    try:
        hours = float(course_row.get('hours_per_week', 8))
    except Exception:
        hours = 8
    if dw and dw > 0:
        return max(1, dw)
    level = str(course_row.get('level','')).lower()
    if 'advanced' in level:
        total = 80
    elif 'intermediate' in level:
        total = 60
    else:
        total = 40
    weeks = ceil(total / max(1, hours))
    return max(1, weeks)

def add_timeline_info(course_row, start_date=None):
    weeks = estimate_weeks(course_row)
    if start_date is None:
        start_date = datetime.utcnow().date()
    end_date = start_date + timedelta(weeks=weeks)
    return {"start_date": str(start_date), "weeks": weeks, "end_date": str(end_date)}

def clean_url(url, provider=None, title=None):
    if not url or str(url).strip().lower() in ['nan', 'none', '']:
        if provider:
            provider_q = re.sub(r'\s+', '+', provider.strip())
            query = re.sub(r'\s+', '+', (title or 'course'))
            return f"https://www.google.com/search?q={provider_q}+{query}"
        return "#"
    url = str(url).strip()
    if not re.match(r'^https?://', url):
        url = "https://" + url
    return url

# ---------------------------
# Embeddings: OpenRouter primary, local / tfidf fallbacks
# ---------------------------
def call_openrouter_embeddings(texts, model="google/gemini-embedding-001", sleep_between=0.05):
    """
    Call OpenRouter embeddings endpoint (preferred remote).
    Expects OPENROUTER_API_KEY in env.
    """
    if isinstance(texts, str):
        texts = [texts]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment.")
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Content-Type":"application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "input": texts}
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Network error calling OpenRouter embeddings: {e}")
    if r.status_code != 200:
        try:
            err = r.json()
            raise RuntimeError(f"OpenRouter embeddings error {r.status_code}: {json.dumps(err)[:1500]}")
        except Exception:
            raise RuntimeError(f"OpenRouter embeddings error {r.status_code}: {r.text[:1500]}")
    try:
        data = r.json()
    except Exception:
        raise RuntimeError("OpenRouter returned non-JSON response for embeddings.")
    vectors = []
    if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
        for item in data['data']:
            vec = item.get('embedding') or item.get('embedding', {}).get('values')
            if vec is None:
                raise RuntimeError("Unexpected OpenRouter embeddings response shape: " + json.dumps(data)[:1000])
            vectors.append(vec)
    else:
        raise RuntimeError("Unexpected OpenRouter embeddings response: " + json.dumps(data)[:1000])
    for i in range(len(vectors)-1):
        time.sleep(sleep_between)
    return vectors

def build_embeddings_and_search(profile_text, df, top_k=50, mode="auto"):
    texts = [
        f"{r['title']} | {r['provider']} | {r.get('skill_tags','')} | {r.get('outcomes','')} | {r.get('level','')}"
        for _, r in df.iterrows()
    ]
    mode = (mode or "auto").lower()

    def try_openrouter():
        all_texts = texts + [profile_text]
        vecs = call_openrouter_embeddings(all_texts, model="google/gemini-embedding-001")
        em = np.vstack(vecs[:-1])
        pvec = np.array(vecs[-1])
        em = em / np.linalg.norm(em, axis=1, keepdims=True)
        pvec = pvec / np.linalg.norm(pvec)
        sims = (em @ pvec).tolist()
        pairs = list(enumerate(sims))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        return pairs_sorted[:top_k], "openrouter"

    def try_local():
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers not installed or failed to import.")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        em = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        em = em / np.linalg.norm(em, axis=1, keepdims=True)
        pvec = model.encode([profile_text], convert_to_numpy=True)[0]
        pvec = pvec / np.linalg.norm(pvec)
        sims = (em @ pvec).tolist()
        pairs = list(enumerate(sims))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        return pairs_sorted[:top_k], "local_embed"

    def try_tfidf():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        vectorizer.fit(texts + [profile_text])
        X = vectorizer.transform(texts)
        p = vectorizer.transform([profile_text])
        sims = cosine_similarity(p, X)[0].tolist()
        pairs = list(enumerate(sims))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        return pairs_sorted[:top_k], "tfidf"

    if mode == "openrouter":
        return try_openrouter()
    elif mode == "local":
        try:
            return try_local()
        except Exception as e:
            st.warning("Local embeddings requested but failed; falling back to TF-IDF. See logs.")
            print("Local embeddings failure:", repr(e))
            return try_tfidf()
    elif mode == "tfidf":
        return try_tfidf()
    else:  # auto
        # prefer OpenRouter -> local -> tfidf
        if os.environ.get("OPENROUTER_API_KEY"):
            try:
                return try_openrouter()
            except Exception as e:
                print("OpenRouter auto attempt failed:", repr(e))
        try:
            return try_local()
        except Exception as e:
            print("Local embedding attempt failed:", repr(e))
        return try_tfidf()

# ---------------------------
# Resume parser (email displayed, phone hidden)
# ---------------------------
def simple_resume_parser(text):
    result = {"email": None, "phone": None, "skills": []}
    m = re.search(r'[\w\.-]+@[\w\.-]+\.\w{2,}', text)
    if m:
        result['email'] = m.group(0)
    phones = []
    for m in re.finditer(r'(\+?\d{1,3}[-.\s]?)?((?:\d{2,4}[-.\s]?){2,4}\d{1,4})', text):
        cand = m.group(0)
        digits = re.sub(r'\D', '', cand)
        if not digits:
            continue
        if len(digits) >= 12 and digits.startswith('20'):
            continue
        if 7 <= len(digits) <= 15:
            if cand.strip().startswith('+'):
                phones.append("+" + digits)
            else:
                phones.append(digits)
    if phones:
        result['phone'] = phones[0]
    skills = []
    for line in text.splitlines():
        if re.search(r'(?i)^\s*(skills|technical skills|technologies|competences)\s*[:\-]', line):
            parts = re.split(r':|-', line, maxsplit=1)
            if len(parts) > 1:
                skills += [s.strip() for s in re.split(r',|;', parts[1]) if s.strip()]
    common_tokens = ['python','sql','react','node','docker','kubernetes','pytorch','tensorflow','excel','tableau','aws','gcp','spark','airflow','java','r','matlab','nlp','pandas','sklearn']
    for token in common_tokens:
        if re.search(r'(?i)\b' + re.escape(token) + r'\b', text):
            skills.append(token)
    result['skills'] = list(dict.fromkeys([s.lower() for s in skills]))
    return result

def parse_resume_file(uploaded_file):
    raw = uploaded_file.read()
    try:
        raw_text = raw.decode('utf-8', errors='ignore')
    except Exception:
        raw_text = str(raw)
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(raw_text)
        nouns = [chunk.text for chunk in doc.noun_chunks][:50]
        parsed = simple_resume_parser(raw_text)
        if parsed['skills']:
            parsed['skills'] = list(dict.fromkeys(parsed['skills'] + nouns))
        else:
            parsed['skills'] = nouns[:10]
        return parsed
    except Exception:
        return simple_resume_parser(raw_text)

# ---------------------------
# Fit normalization function (prevents the 90-everywhere problem)
# ---------------------------
def normalize_fit(sim, level_bonus, prereq_penalty, popularity, sim_type="openrouter"):
    """
    Generate a human-friendly fit score spread across ~70-100
    Inputs:
      - sim: cosine similarity in [0,1]
      - level_bonus: numerical bonus (0/5/10)
      - prereq_penalty: numerical penalty
      - popularity: numeric score (1-10 typically)
      - sim_type: unused for now but kept for future calibration
    Output: score ∈ [70, 100] (float)
    """
    # Base: amplify similarity so it matters
    base_signal = sim * 85.0  # similarity -> up to 85
    # Level bonus helps push good matches higher
    lvl = float(level_bonus) * 2.5
    # Popularity is a small nudge
    pop = float(popularity) * 1.2
    # Penalty harms score
    pen = float(prereq_penalty) * 4.0

    raw_score = base_signal + lvl + pop - pen

    # Add a tiny deterministic tie-break using popularity and sim
    raw_score += (sim * 3.0) + (0.1 * (popularity % 3))

    # Map raw_score into [70, 100] while preserving spread
    # First clamp reasonable raw bounds
    raw_score = max(0.0, min(110.0, raw_score))
    # Normalize: assume raw_score roughly in [30,110] => map to [70,100]
    min_raw, max_raw = 30.0, 110.0
    scaled = 70.0 + ((raw_score - min_raw) / (max_raw - min_raw)) * 30.0
    scaled = max(70.0, min(100.0, scaled))
    return round(scaled, 1)

# ---------------------------
# Sidebar: profile, embedding mode, resume, filters
# ---------------------------
st.sidebar.header("Profile")
name = st.sidebar.text_input("Your name", value="Anonymous")
education = st.sidebar.selectbox("Education level", ["High School", "BA/BSc", "BTech/BEng", "MSc/PhD", "Other"])
major = st.sidebar.text_input("Major / Degree (optional)", value="")
tech_skills_text = st.sidebar.text_area("Technical skills (comma separated)", value="Python (basic), SQL (basic)", height=100)
soft_skills_text = st.sidebar.text_area("Soft skills (comma separated)", value="communication, analytical", height=80)
target_domain = st.sidebar.text_input("Target domain (optional)", value="Data Analytics")
target_job = st.sidebar.text_input("Target job (optional)", value="Data Analyst")
user_level_guess = st.sidebar.selectbox("Self-assessed level", ["beginner","beginner-intermediate","intermediate","advanced"])
preferred_hours = st.sidebar.selectbox("Hours you can study per week", [4,6,8,10,12], index=2)

st.sidebar.markdown("---")
st.sidebar.subheader("Embedding mode (toggle for demos)")
emb_mode = st.sidebar.radio("Choose embedding mode:", ("Auto", "OpenRouter (preferred)", "Local embeddings", "Offline TF-IDF"), index=0)
st.sidebar.caption("Auto: OpenRouter if server has key → otherwise local → otherwise offline TF-IDF.")

st.sidebar.markdown("---")
mode_cost = st.sidebar.selectbox("Cost mode", ["Free / Audit", "Paid", "All"])
program_type_filter = st.sidebar.multiselect("Program types", ["college_program","specialization","certification","curated_path","program","course"], default=["certification","specialization","curated_path","college_program","program","course"])

st.sidebar.markdown("---")
st.sidebar.subheader("Upload resume (optional)")
uploaded_resume = st.sidebar.file_uploader("Upload resume (PDF/TXT) — parsed locally only", type=["pdf","txt","docx"])
if uploaded_resume:
    try:
        parsed = parse_resume_file(uploaded_resume)
        if parsed.get("email"):
            st.sidebar.write("Email detected:", parsed["email"])
        # phone intentionally not displayed
        if parsed.get("skills"):
            existing = [s.strip() for s in tech_skills_text.split(",") if s.strip()]
            merged = existing + [s for s in parsed["skills"] if s not in existing]
            tech_skills_text = ", ".join(merged)
            st.sidebar.success("Skills auto-detected and added.")
    except Exception:
        st.sidebar.warning("Resume parsing failed (fallback).")

st.sidebar.markdown("---")
st.sidebar.subheader("Samples")
sample_choice = st.sidebar.selectbox("Sample persona", ["None","A - Beginner CS grad","B - Frontend→ML","C - BA→Product Analytics","D - Advanced ML→MLOps","E - Full-stack dev"])
if st.sidebar.button("Load sample profile"):
    try:
        samples = json.load(open("sample_profiles.json"))
        mapping = {"A - Beginner CS grad":"A","B - Frontend→ML":"B","C - BA→ProductAnalytics":"C","D - Advanced ML→MLOps":"D","E - Full-stack dev":"E"}
        if sample_choice != "None":
            chosen = [s for s in samples if s["id"] == mapping[sample_choice]][0]
            name = chosen["name"]
            education = chosen["education"]
            tech_skills_text = ", ".join(chosen["tech_skills"])
            soft_skills_text = ", ".join(chosen["soft_skills"])
            target_domain = chosen.get("target_domain", target_domain)
            target_job = chosen.get("target_job", target_job)
            preferred_hours = chosen.get("preferred_hours", preferred_hours)
            user_level_guess = chosen.get("user_level_guess", user_level_guess)
            st.sidebar.success(f"Loaded sample: {chosen['name']}")
    except Exception:
        st.sidebar.error("Could not load sample_profiles.json. Ensure it exists and is valid JSON.")

submit = st.sidebar.button("Recommend")
st.sidebar.markdown("---")
st.sidebar.caption(f"Built: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ---------------------------
# Main UI header
# ---------------------------
st.title("Learning Recommender — Programs, Certifications & Paths")
st.write("Personalized recommendations based on education, skills, and career goals.")

k1,k2,k3,k4 = st.columns(4)
k1.metric("Catalog size", "80+")
k2.metric("Mode", mode_cost)
k3.metric("Target job", target_job or "—")
k4.metric("Level", user_level_guess)

if not submit:
    st.info("Fill the profile in the sidebar and click Recommend.")
    st.stop()

# ---------------------------
# Pipeline: load data, build profile, search
# ---------------------------
df = load_catalog('courses_extended.csv')
tech_skills = [s.strip() for s in tech_skills_text.split(',') if s.strip()]
soft_skills = [s.strip() for s in soft_skills_text.split(',') if s.strip()]
profile_text = f"{education}. Major: {major}. Skills: {', '.join(tech_skills)}. Soft: {', '.join(soft_skills)}. Goal: {target_domain}. Job: {target_job}. Level: {user_level_guess}"

_mode_map = {"Auto":"auto", "OpenRouter (preferred)":"openrouter", "Local embeddings":"local", "Offline TF-IDF":"tfidf"}
selected_mode = _mode_map.get(emb_mode, "auto")

try:
    sims_with_type, sim_type = build_embeddings_and_search(profile_text, df, top_k=80, mode=selected_mode)
    sims = sims_with_type
except Exception as e:
    st.error("Search error — check server logs for details.")
    st.code(traceback.format_exc().splitlines()[-1])
    st.stop()

# ---------------------------
# Score & collect candidates
# ---------------------------
candidates = []
for idx, sim in sims:
    row = df.iloc[idx].to_dict()
    if row.get('program_type') not in program_type_filter:
        continue
    is_free_flag = str(row.get('is_free','yes')).lower() in ['yes','true','1']
    if mode_cost == "Free / Audit" and not is_free_flag:
        continue
    if mode_cost == "Paid" and is_free_flag:
        continue

    prereq_pen, missing_prereqs = simple_prereq_penalty(tech_skills, row.get('prerequisites','none'))
    level_bonus = level_bonus_score(user_level_guess, row.get('level','beginner'))
    popularity = float(row.get('popularity_score',5))

    # use similarity (sim) directly in normalize_fit to get spread
    displayed_fit = normalize_fit(sim, level_bonus, prereq_pen, popularity, sim_type=sim_type)

    skill_sim_score = round((sim or 0.0) * (200.0 if sim_type != 'tfidf' else 600.0), 2)

    justification = deterministic_rationale(row, profile_text, ", ".join(tech_skills[:2]) or row.get('skill_tags',''))
    steps = []
    if row.get('first_project'):
        steps.append(f"Build: {row.get('first_project')}")
    else:
        skills_text = " ".join([str(row.get('skill_tags','')), str(row.get('outcomes','')), str(row.get('title',''))]).lower()
        lead = None
        for tok in ['python','sql','ml','mlops','pytorch','docker','kubernetes','react','node','excel','tableau','nlp','airflow','testing','devops','cloud','pandas','tensorflow','spark','etl']:
            if tok in skills_text:
                lead = tok
                break
        if not lead:
            lead = (row.get('outcomes','').split(',')[0] if row.get('outcomes') else 'relevant skills')
        steps.append(f"Build a portfolio project using {lead}.")
        if 'sql' in lead:
            steps.append("Practice: solve SQL problems and write queries.")
        else:
            steps.append("Practice: turn course exercises into a polished deliverable.")
        if 'coursera' in str(row.get('provider','')).lower() or 'edx' in str(row.get('provider','')).lower():
            steps.append("Certify: obtain course certificate or archive artifacts.")
        else:
            steps.append("Document: write a short case study of your project.")

    timeline = add_timeline_info(row, start_date=datetime.utcnow().date())

    candidates.append({
        "id": row.get('id'),
        "title": row.get('title'),
        "provider": row.get('provider'),
        "program_type": row.get('program_type'),
        "outcomes": row.get('outcomes'),
        "skill_tags": row.get('skill_tags'),
        "level": row.get('level'),
        "fit_raw_sim": sim,
        "fit": displayed_fit,
        "skill_sim_score": skill_sim_score,
        "prereq_penalty": round(prereq_pen,2),
        "missing_prereqs": missing_prereqs,
        "level_bonus": round(level_bonus,2),
        "popularity": popularity,
        "rationale": justification,
        "first_project": row.get('first_project'),
        "hours_per_week": row.get('hours_per_week'),
        "cost_estimate_usd": row.get('cost_estimate_usd'),
        "mode": row.get('mode'),
        "credential": row.get('credential'),
        "free_link": row.get('free_link'),
        "link": row.get('link') if 'link' in row else row.get('free_link'),
        "is_free": row.get('is_free'),
        "timeline": timeline,
        "why_help_for_job": justification,
        "next_steps_for_job": steps
    })

def final_score_adjust(c):
    adj = c['fit']
    if user_level_guess == 'beginner' and c['level'] in ['intermediate','advanced'] and c['missing_prereqs']:
        adj -= 8.0
    return adj

candidates_sorted = sorted(candidates, key=lambda x: final_score_adjust(x), reverse=True)
top = candidates_sorted[:20]
ordered_top = build_ordered_path(top, df)
ordered_top6 = ordered_top[:6]

# ---------------------------
# Render UI results
# ---------------------------
st.markdown("## Top recommendations")
cols = st.columns(3)
for i, r in enumerate(ordered_top6):
    with cols[i % 3]:
        free_tag = " (Free)" if str(r.get('is_free','yes')).lower() in ['yes','true','1'] else ""
        st.markdown(f"**{r['title']}** — *{r['provider']}*{free_tag}")
        st.markdown(f"**Type:** {r.get('program_type','')}, **Mode:** {r.get('mode','')}, **Credential:** {r.get('credential','')}")
        st.metric(label="Fit score", value=f"{r['fit']}/100")
        st.progress(int(max(70, min(100, r['fit']))))
        st.write(r.get('rationale'))
        st.write(f"**Outcomes:** {r.get('outcomes','')}")
        st.write(f"**Hours / week:** {r.get('hours_per_week')} • **Est. duration (weeks):** {r.get('timeline',{}).get('weeks')}")
        st.write(f"**Estimated end date:** {r.get('timeline',{}).get('end_date')}")
        if r.get('first_project'):
            st.info(f"Quick build: {r.get('first_project')}")
        st.markdown(f"**Why this helps for _{target_job}_**")
        st.write(r.get('why_help_for_job',''))
        st.markdown("**Next steps to attain the job skills**")
        for step in r.get('next_steps_for_job', []):
            st.write(f"- {step}")
        link_to_use = r.get('link') or r.get('free_link') or ""
        cleaned = clean_url(link_to_use, provider=r.get('provider'), title=r.get('title'))
        if cleaned == "#":
            st.write("Enroll / Free link: Not available")
        else:
            st.markdown(f"[Enroll / Free link]({cleaned})")
        with st.expander("Score breakdown"):
            st.write(f"- sim_backend: **{sim_type}**")
            st.write(f"- sim (cosine): **{r.get('fit_raw_sim')}**")
            st.write(f"- displayed_fit: **{r['fit']}**")
            st.write(f"- skill_sim_score: **{r['skill_sim_score']}**")
            st.write(f"- prereq_penalty: **{r['prereq_penalty']}**")
            st.write(f"- missing_prereqs: **{', '.join(r['missing_prereqs']) if r['missing_prereqs'] else 'none'}**")
            st.write(f"- level_bonus: **{r['level_bonus']}**")
            st.write(f"- popularity: **{r['popularity']}**")

st.markdown("## Suggested learning path (ordered)")
for idx, step in enumerate(ordered_top, start=1):
    st.write(f"{idx}. **{step['title']}** — {step['program_type']} • Fit: {step['fit']} • Est. weeks: {step['timeline'].get('weeks')} • End: {step['timeline'].get('end_date')}")

# Career roadmap generator
def generate_career_roadmap(target_job, ordered_courses, preferred_hours):
    if not target_job:
        target_job = "Target Role"
    roadmap = {"short_term": [], "mid_term": [], "long_term": []}
    for c in ordered_courses:
        weeks = c.get('timeline',{}).get('weeks', 0)
        if weeks <= 8 and len(roadmap['short_term']) < 3:
            roadmap['short_term'].append(c)
    for c in ordered_courses:
        weeks = c.get('timeline',{}).get('weeks', 0)
        if 8 < weeks <= 24 and len(roadmap['mid_term']) < 3 and c not in roadmap['short_term']:
            roadmap['mid_term'].append(c)
    for c in ordered_courses:
        if c not in roadmap['short_term'] and c not in roadmap['mid_term'] and len(roadmap['long_term']) < 3:
            roadmap['long_term'].append(c)
    def enrich(entry, phase):
        weeks = entry.get('timeline',{}).get('weeks', 4)
        hours = int(preferred_hours) if preferred_hours else 6
        est_total_hours = weeks * hours
        project = entry.get('first_project') or f"Project: build a practical artifact using {entry.get('skill_tags','').split(',')[0] or entry.get('outcomes','').split(',')[0]}"
        return {
            "id": entry.get('id'),
            "title": entry.get('title'),
            "provider": entry.get('provider'),
            "weeks": weeks,
            "hours_per_week": hours,
            "est_total_hours": est_total_hours,
            "project": project,
            "link": entry.get('link') or entry.get('free_link') or ""
        }
    roadmap['short_term'] = [enrich(c,"short") for c in roadmap['short_term']]
    roadmap['mid_term'] = [enrich(c,"mid") for c in roadmap['mid_term']]
    roadmap['long_term'] = [enrich(c,"long") for c in roadmap['long_term']]
    return roadmap

roadmap = generate_career_roadmap(target_job, ordered_top, preferred_hours)
st.markdown("## Career roadmap (tailored)")
st.write(f"**Target role:** {target_job or 'Target Role'}")
st.write("### Short-term (0-3 months) — quick wins & projects")
if roadmap['short_term']:
    for r in roadmap['short_term']:
        st.write(f"- **{r['title']}** ({r['provider']}) — Est {r['weeks']} weeks • {r['hours_per_week']} hrs/week • Project: {r['project']} • [Link]({clean_url(r['link'])})")
else:
    st.write("- No short-term items found.")

st.write("### Mid-term (3-6 months) — deepen skills & portfolio")
if roadmap['mid_term']:
    for r in roadmap['mid_term']:
        st.write(f"- **{r['title']}** ({r['provider']}) — Est {r['weeks']} weeks • {r['hours_per_week']} hrs/week • Project: {r['project']} • [Link]({clean_url(r['link'])})")
else:
    st.write("- No mid-term items found.")

st.write("### Long-term (6-12+ months) — specialization & capstones")
if roadmap['long_term']:
    for r in roadmap['long_term']:
        st.write(f"- **{r['title']}** ({r['provider']}) — Est {r['weeks']} weeks • {r['hours_per_week']} hrs/week • Project: {r['project']} • [Link]({clean_url(r['link'])})")
else:
    st.write("- No long-term items found.")

# Timeline summary
st.markdown("## Timeline summary")
short_term = [c for c in ordered_top if c['timeline']['weeks'] <= 8][:3]
long_term = [c for c in ordered_top if c['timeline']['weeks'] > 8][:5]
st.subheader("Short-term (next 1–3 months)")
if short_term:
    for s in short_term:
        st.write(f"- **{s['title']}** — Est. {s['timeline']['weeks']} weeks, Build: {s.get('first_project')}")
else:
    st.write("- No short-term items found.")
st.subheader("Long-term (3–12 months)")
if long_term:
    for l in long_term:
        st.write(f"- **{l['title']}** — Est. {l['timeline']['weeks']} weeks")
else:
    st.write("- No long-term items found.")

export = {
    "profile": {
        "name": name,
        "education": education,
        "major": major,
        "tech_skills": tech_skills,
        "soft_skills": soft_skills,
        "goal": target_domain,
        "level": user_level_guess,
        "target_job": target_job,
        "preferred_hours_per_week": preferred_hours
    },
    "generated_at": datetime.utcnow().isoformat(),
    "recommendations": ordered_top,
    "roadmap": roadmap
}
st.download_button("Download full recommendations (JSON)", data=json.dumps(export, indent=2), file_name="recommendations_full.json", mime="application/json")
st.markdown("---")
st.caption("Server-side embeddings: set OPENROUTER_API_KEY in the environment to enable OpenRouter (preferred). If not set, the app falls back to local sentence-transformers or TF-IDF.")

# End of file
