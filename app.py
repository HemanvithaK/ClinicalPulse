import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ClinicalPulse",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: #1e2130;
        border: 1px solid #6366f1;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #e2e8f0;
        line-height: 1.7;
    }
    .citation-box {
        background: #151720;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    .citation-item {
        color: #818cf8;
        font-size: 0.82rem;
        padding: 2px 0;
    }
    .stat-card {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
    }
    .stat-label {
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 2px;
    }
    .confidence-high { color: #4ade80; }
    .confidence-med  { color: #facc15; }
    .confidence-low  { color: #f87171; }
</style>
""", unsafe_allow_html=True)


def query_api(question: str) -> dict:
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": question, "voice_response": False},
            timeout=60,
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


st.markdown('<div class="hero-title">🧬 ClinicalPulse</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Real-time clinical trial intelligence · '
    'GraphRAG + Hybrid Retrieval + Voice AI</div>',
    unsafe_allow_html=True,
)

api_ok = check_api_health()
if api_ok:
    st.success("API connected — ClinicalPulse is ready", icon="✅")
else:
    st.error("API not reachable — make sure the API server is running on port 8000", icon="🔴")
    st.stop()

st.divider()

def get_live_stats() -> dict:
    try:
        from qdrant_client import QdrantClient
        from neo4j import GraphDatabase
        
        qdrant = QdrantClient(host="localhost", port=6333)
        vectors = qdrant.get_collection("clinical_trials").points_count
        
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "clinical123")
        )
        with driver.session() as session:
            trials = session.run(
                "MATCH (t:Trial) RETURN count(t) as c"
            ).single()["c"]
            drugs = session.run(
                "MATCH (d:Drug) RETURN count(d) as c"
            ).single()["c"]
            conditions = session.run(
                "MATCH (c:Condition) RETURN count(c) as c"
            ).single()["c"]
        driver.close()
        
        return {
            "trials": trials,
            "vectors": vectors,
            "drugs": drugs,
            "conditions": conditions,
        }
    except Exception:
        return {"trials": 0, "vectors": 0, "drugs": 0, "conditions": 0}


def get_eval_scores() -> dict:
    import json
    import pathlib
    results_file = pathlib.Path("eval/eval_results.json")
    if results_file.exists():
        data = json.loads(results_file.read_text())
        return {
            "faithfulness": data.get("avg_faithfulness", 0),
            "relevance": data.get("avg_relevance", 0),
            "completeness": data.get("avg_completeness", 0),
            "composite": data.get("avg_composite", 0),
        }
    return {
        "faithfulness": 0,
        "relevance": 0,
        "completeness": 0,
        "composite": 0,
    }

stats = get_live_stats()
eval_scores = get_eval_scores()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{stats['trials']:,}</div>
        <div class="stat-label">Trials in graph</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{stats['conditions']:,}</div>
        <div class="stat-label">Conditions indexed</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{eval_scores['composite']:.2f}</div>
        <div class="stat-label">Eval composite score</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-number">{stats['drugs']:,}</div>
        <div class="stat-label">Drugs in graph</div>
    </div>""", unsafe_allow_html=True)

st.divider()

tab1, tab2 = st.tabs(["🔍 Text Search", "🎙️ Voice Search"])

with tab1:
    left, right = st.columns([1.2, 1], gap="large")

    with left:
        st.markdown("### Ask ClinicalPulse")
        st.markdown("**Try an example:**")
        examples = [
            "What Phase 3 breast cancer trials use tamoxifen?",
            "Which diabetes trials are currently recruiting?",
            "What Alzheimers trials focus on memory improvement?",
            "Which trials are sponsored by the Institute of Cancer Research?",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state["query_input"] = ex

        st.markdown("")

        query = st.text_area(
            "Your question",
            value=st.session_state.get("query_input", ""),
            placeholder="e.g. What Phase 3 trials are recruiting for lung cancer?",
            height=100,
            key="query_input",
        )

        ask_col, clear_col = st.columns([3, 1])
        ask_clicked = ask_col.button(
            "🔍 Search trials",
            type="primary",
            use_container_width=True,
        )
        if clear_col.button("Clear", use_container_width=True):
            st.session_state["query_input"] = ""
            st.session_state.pop("last_result", None)
            st.rerun()

        if ask_clicked and query.strip():
            with st.spinner("Searching trials across Neo4j, Qdrant, and BM25..."):
                result = query_api(query.strip())
                st.session_state["last_result"] = result

        if "last_result" in st.session_state:
            result = st.session_state["last_result"]

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                answer = result.get("answer", "")
                citations = result.get("citations", [])
                confidence = result.get("confidence_score", 0)
                flags = result.get("hallucination_flags", [])

                if confidence >= 0.9:
                    conf_class = "confidence-high"
                    conf_label = "High confidence"
                elif confidence >= 0.7:
                    conf_class = "confidence-med"
                    conf_label = "Medium confidence"
                else:
                    conf_class = "confidence-low"
                    conf_label = "Low confidence"

                st.markdown(
                    f'<span class="{conf_class}">● {conf_label} ({confidence:.0%})</span>',
                    unsafe_allow_html=True,
                )

                if flags:
                    st.warning(f"⚠️ {len(flags)} claim(s) could not be fully verified")

                st.markdown(
                    f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True,
                )

                if citations:
                    st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                    st.markdown("**Sources**")
                    for c in citations:
                        st.markdown(
                            f'<div class="citation-item">🔗 {c}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### How it works")

        with st.expander("🔍 Hybrid retrieval", expanded=True):
            st.markdown("""
Your query hits **3 search sources simultaneously:**
- **Vector search** (Qdrant) — semantic similarity
- **BM25** — exact keyword matching
- **Graph traversal** (Neo4j) — relationship-based search

Results are merged with **Reciprocal Rank Fusion**, then reranked by a cross-encoder model.
            """)

        with st.expander("🤖 LangGraph agent", expanded=True):
            st.markdown("""
A 4-node stateful graph:
1. **Planner** — classifies intent, extracts keywords
2. **Retriever** — runs hybrid search
3. **Synthesizer** — Claude generates cited answer
4. **Guardrails** — checks every claim against sources

If hallucinations are detected, the agent **automatically regenerates** up to 2 times.
            """)

        with st.expander("📊 Evaluation scores"):
            scores = get_eval_scores()
            st.markdown(f"""Evaluated with LLM-as-Judge (Claude):

| Metric | Score |
|---|---|
| Faithfulness | {scores['faithfulness']:.3f} |
| Relevance | {scores['relevance']:.3f} |
| Completeness | {scores['completeness']:.3f} |
| **Composite** | **{scores['composite']:.3f}** |
    """)
            

        with st.expander("🧬 Data source"):
            st.markdown("""
- **Source:** ClinicalTrials.gov (NIH)
- **Trials indexed:** 150+ across cancer, diabetes, Alzheimers
- **Graph nodes:** Trial, Condition, Drug, Sponsor, Location
- **Vector store:** Qdrant (1536-dim, text-embedding-3-small)
- **Update frequency:** Every 6 hours via APScheduler
            """)

with tab2:
    st.markdown("### 🎙️ Voice Search")
    st.markdown(
        "Type your query and ClinicalPulse will **speak the answer back to you** "
        "using the full voice pipeline — LangGraph agent → OpenAI TTS."
    )

    voice_query = st.text_input(
        "Your question",
        placeholder="e.g. What Alzheimers trials focus on memory improvement?",
        key="voice_query",
    )

    if st.button("🎙️ Get Voice Answer", type="primary", use_container_width=True):
        if voice_query.strip():

            with st.spinner("Running agent..."):
                try:
                    text_result = query_api(voice_query.strip())
                    answer = text_result.get("answer", "")
                    citations = text_result.get("citations", [])
                    confidence = text_result.get("confidence_score", 0)
                    flags = text_result.get("hallucination_flags", [])

                    if confidence >= 0.9:
                        conf_class = "confidence-high"
                        conf_label = "High confidence"
                    elif confidence >= 0.7:
                        conf_class = "confidence-med"
                        conf_label = "Medium confidence"
                    else:
                        conf_class = "confidence-low"
                        conf_label = "Low confidence"

                    st.markdown(
                        f'<span class="{conf_class}">● {conf_label} ({confidence:.0%})</span>',
                        unsafe_allow_html=True,
                    )

                    if flags:
                        st.warning(f"⚠️ {len(flags)} claim(s) could not be fully verified")

                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True,
                    )

                    if citations:
                        st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                        st.markdown("**Sources**")
                        for c in citations:
                            st.markdown(
                                f'<div class="citation-item">🔗 {c}</div>',
                                unsafe_allow_html=True,
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.stop()

            with st.spinner("Synthesizing voice response..."):
                try:
                    audio_response = requests.post(
                        f"{API_URL}/query/audio",
                        json={"query": voice_query.strip(), "voice_response": True},
                        timeout=60,
                    )
                    if audio_response.status_code == 200:
                        audio_bytes = audio_response.content
                        st.success(
                            f"🔊 Voice response ready — "
                            f"{len(audio_bytes):,} bytes via OpenAI TTS"
                        )
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    else:
                        st.error(f"Audio synthesis failed: {audio_response.status_code}")
                except Exception as e:
                    st.error(f"TTS error: {e}")
        else:
            st.warning("Please enter a question first")

    st.divider()
    st.markdown("#### How the voice pipeline works")
    st.code("""
Your text input
       |
       v
LangGraph agent  ->  cited answer
       |
       v
OpenAI TTS (tts-1, alloy voice)  ->  MP3 audio
       |
       v
Played directly in your browser
    """, language=None)

st.divider()
st.markdown(
    '<div style="text-align:center; color:#475569; font-size:0.8rem;">'
    'ClinicalPulse · LangGraph · Neo4j · Qdrant · Claude API · Whisper · OpenAI TTS · GCP Cloud Run'
    '</div>',
    unsafe_allow_html=True,
)