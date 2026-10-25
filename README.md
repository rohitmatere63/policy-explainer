# Policy Explainer Lite (Streamlit + Sarvam AI)

Upload an English health policy PDF, ask questions in Hindi/Tamil/Bengali/etc., get plain-language answers with
verified clause citations and a highlighted PDF page. Weak answers are flagged instead of guessed.

## Run locally (Windows PowerShell, your own venv)
    uv venv .venv --python 3.11
    .venv\Scripts\Activate.ps1
    uv pip install -r requirements.txt
    copy .env.example .env          # put SARVAM_API_KEY inside
    streamlit run app.py

## Deploy free (about 5 minutes)
1. Push this folder to a GitHub repo (`.env` is git-ignored).
2. share.streamlit.io -> New app -> pick repo, main file `app.py`.
3. App settings -> Secrets -> paste `SARVAM_API_KEY = "your_key"`.

Files: `app.py` (UI) · `engine.py` (parse, search, answer, verify) · `sarvam.py` (API client).
Model names are env-configurable (see top of `sarvam.py`); check docs.sarvam.ai if you get a 404.
