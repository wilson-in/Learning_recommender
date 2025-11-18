---
title: AI Learning Path Recommender
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.40.0"
app_file: app.py
pinned: false
---

# 🧠 AI Learning Path Recommender
### Personalized Courses, Certifications & College Programs — Offline-First, Explainable & Free

[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-green?style=flat-square)](#)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-orange?style=flat-square)](#)

---

## ⭐ Overview

Learners today face too many options — courses, certifications, bootcamps, YouTube playlists, specializations, and even full college programs.

Choosing the *right next step* is hard.

This project is a **full AI system** that analyzes a user's:
- Education  
- Technical skills  
- Soft skills  
- Target domains  
- Target job role  
- Study availability  

…and produces:
- 🎯 Ranked course/program recommendations  
- 🧭 Short-term & long-term learning timeline  
- 🗺 Skill-gap analysis  
- 💼 Job-targeted rationales  
- 🔗 Direct enrollment links (free/paid)  
- 📦 JSON export  
- 🧩 Optional LLM-enhanced explanations via OpenRouter  

The entire system works:
- **Offline-first** (TF-IDF & local embeddings available)  
- **Online-enhanced** (Gemini 2.0 embeddings + OpenRouter LLM optional)  
- **100% free to run**  
- **Reproducible** (CSV catalog + deterministic scoring + tests)  

---

## 🚀 Features

- **Profile intake**
  - Education, major, goals, tech skills, soft skills  
  - Target job role  
  - Study hours/week  
  - Resume upload (local parsing)  

- **Course Catalog System**
  - 25–50 curated items  
  - Title, provider, duration, cost, level, prerequisites, skill tags, links  

- **Matching Engine**
  - Gemini embeddings (optional, server-side)  
  - Local SentenceTransformers (optional)  
  - TF-IDF fallback (offline guaranteed)  
  - Deterministic scoring: prerequisites, level, popularity, skill similarity  

- **Ranking**
  - Fit Score (0–100)  
  - Beginner gating (prevents recommending advanced courses too early)  

- **Timeline Engine**
  - Automatic week estimates  
  - Estimated end date  
  - Short-term (1–3 months)  
  - Long-term (3–12 months)  

- **Explainers**
  - Deterministic “why this helps”  
  - Optional LLM reasoning (via OpenRouter: Llama 3.1, DeepSeek R1, etc.)  

- **Outputs**
  - UI cards  
  - Downloadable JSON  
  - Graph visualization (optional)  

---

## 📁 Project Structure

learning-recommender/
│
├── app.py # Main Streamlit UI
├── courses.csv # Course metadata (offline catalog)
├── sample_profiles.json # 5 sample learner personas
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
├── INSTALL.md
├── CONTRIBUTING.md
│
├── tests/
│ └── test_compute_fit.py # Deterministic scoring unit tests
│
└── .streamlit/
└── example.secrets.toml # Safe template for secret keys


---

## 🧪 Local Setup

1. **Clone repo**


git clone https://github.com/yourusername/learning-recommender.git

cd learning-recommender


2. **Create environment**


python -m venv .venv
source .venv/bin/activate # Mac/Linux
..venv\Scripts\activate # Windows


3. **Install packages**


pip install -r requirements.txt


4. **Run tests**


pytest -q


5. **Run app**


streamlit run app.py