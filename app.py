import base64
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

import engine
import sarvam as sv

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

LANGS = {"English": "en-IN", "हिन्दी (Hindi)": "hi-IN", "বাংলা (Bengali)": "bn-IN", "தமிழ் (Tamil)": "ta-IN",
         "తెలుగు (Telugu)": "te-IN", "ಕನ್ನಡ (Kannada)": "kn-IN", "മലയാളം (Malayalam)": "ml-IN",
         "मराठी (Marathi)": "mr-IN", "ગુજરાતી (Gujarati)": "gu-IN", "ਪੰਜਾਬੀ (Punjabi)": "pa-IN", "ଓଡ଼ିଆ (Odia)": "od-IN"}

sv._key()  # read the API key on the main thread before worker threads start
st.set_page_config(page_title="Policy Explainer", page_icon="🛡️", layout="wide")

with st.sidebar:
    st.title("🛡️ Policy Explainer")
    lang = LANGS[st.selectbox("Language / भाषा", list(LANGS))]
    st.session_state.debug = st.checkbox("Show debug info")
    up = st.file_uploader("Upload health policy (PDF)", type="pdf")
    if up and st.session_state.get("fname") != up.name:
        with st.spinner("Reading policy..."):
            st.session_state.policy = engine.Policy(up.getvalue())
            st.session_state.fname, st.session_state.chat, st.session_state.summary = up.name, [], {}
    pol: engine.Policy | None = st.session_state.get("policy")
    if pol:
        st.success(f"{up.name if up else ''} · {pol.n_pages} pages · {len(pol.chunks)} clauses")
        if pol.low_quality:
            st.warning("Low extraction quality (scanned or unusual layout). Answers may be unreliable.")

try:
    st.info(sv.translate(engine.DISCLAIMER, lang))
except Exception as e:  # show the real Sarvam error (Streamlit Cloud hides uncaught ones)
    st.info(engine.DISCLAIMER)
    st.error(f"Sarvam translate failed: {e}")
if not pol:
    st.write("👈 Upload a health insurance policy PDF to begin.")
    st.stop()


def show(a: dict, key: str):
    for m in a["flags"]:
        st.warning(m)
    if a["short"]:
        st.markdown(f"### {a['short']}")
    for s in a["statements"]:
        st.write(f"• {s['text']}  `{', '.join(s['cites'])}`")
    if a["conditions"]:
        st.markdown("**It depends on:**")
        for c in a["conditions"]:
            st.write(f"– {c}")
    if a["statements"]:
        st.caption(f"Confidence: {a['confidence']:.0%}")
        if st.button("🔊 Listen", key=f"tts-{key}"):
            audio = sv.text_to_speech(a["short"] or a["statements"][0]["text"], lang)
            if audio:
                st.audio(base64.b64decode(audio), format="audio/wav")
    if st.session_state.get("debug"):
        st.json(a.get("debug", {}))
    for i, c in enumerate(a["citations"]):
        with st.expander(f"Page {c['page']} · {c['section']} · clause {c['clause'] or '-'}"):
            st.caption(f"“{c['quote']}”")
            st.image(pol.page_png(c["page"], c["quote"]))


def ask(question_local: str) -> dict:
    q_en = sv.translate(question_local, "en-IN", source=lang) if lang != "en-IN" else question_local
    return engine.answer(pol, q_en, lang)


tab1, tab2 = st.tabs(["Coverage summary", "Ask a question"])

with tab1:
    if st.button("Generate key terms summary", type="primary"):
        try:
            with st.spinner("Reading the policy (about 20-30 seconds)..."):
                items = list(engine.SUMMARY_QUESTIONS.items())
                with ThreadPoolExecutor(4) as ex:
                    results = list(ex.map(lambda kv: engine.answer(pol, kv[1], lang), items))
                st.session_state.summary = {k: r for (k, _), r in zip(items, results)}
        except Exception as e:
            st.error(f"Sarvam error: {e}")
    cols = st.columns(2)
    for i, (label, a) in enumerate(st.session_state.get("summary", {}).items()):
        with cols[i % 2], st.container(border=True):
            st.subheader(sv.translate(label, lang))
            show(a, f"sum{i}")

with tab2:
    for i, m in enumerate(st.session_state.get("chat", [])):
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["content"])
            else:
                show(m["content"], f"chat{i}")
    if q := st.chat_input("Ask about your policy in your language..."):
        st.session_state.chat.append({"role": "user", "content": q})
        try:
            with st.spinner("Searching the policy..."):
                st.session_state.chat.append({"role": "assistant", "content": ask(q)})
            st.rerun()
        except Exception as e:
            st.session_state.chat.pop()
            st.error(f"Sarvam error: {e}")
