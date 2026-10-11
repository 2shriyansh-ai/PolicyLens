from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st

from policylens.answering import answer_question
from policylens.chunking import fixed_size_chunks, structure_aware_chunks
from policylens.config import EmbeddingSettings, LLMSettings
from policylens.documents import DocumentReadError, extract_pages, extract_uploaded_pdf
from policylens.evaluation import calibrate_threshold, evaluate, load_cases, summarize
from policylens.experiments import choose_threshold, compare_chunking, threshold_sweep
from policylens.failures import add_failure_categories, failure_summary
from policylens.grading import build_manual_grading_rows
from policylens.llm import LocalTransformersGenerator, OpenAICompatibleGenerator
from policylens.retrieval import HashingEmbedder, OpenAICompatibleEmbedder, PolicyIndex, SentenceTransformerEmbedder


st.set_page_config(page_title="PolicyLens", page_icon="🔎", layout="wide")
st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem;}
      [data-testid="stMetric"] {border: 1px solid #e4e7ec; border-radius: 12px; padding: 14px;}
      .evidence-card {border-left: 4px solid #2563eb; padding: .7rem 1rem; background: #f8fafc; border-radius: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def make_index(serialized_pages: tuple, chunking: str, embedder_name: str) -> PolicyIndex:
    from policylens.models import Page

    pages = [Page(document=item[0], number=item[1], text=item[2]) for item in serialized_pages]
    chunks = fixed_size_chunks(pages) if chunking == "Fixed-size" else structure_aware_chunks(pages)
    embedder = HashingEmbedder()
    if embedder_name == "Sentence Transformer":
        embedder = SentenceTransformerEmbedder()
    elif embedder_name == "API embeddings":
        settings = EmbeddingSettings.from_environment()
        if not settings:
            raise RuntimeError("Set POLICYLENS_EMBEDDING_API_KEY before using API embeddings.")
        embedder = OpenAICompatibleEmbedder(settings)
    return PolicyIndex(chunks, embedder)


@st.cache_resource(show_spinner=False)
def local_generator() -> LocalTransformersGenerator:
    return LocalTransformersGenerator()


def load_demo_pages():
    benchmark = ROOT / "data" / "benchmark_policies"
    if benchmark.exists():
        return [
            page
            for path in sorted(benchmark.glob("*.pdf"))
            for page in extract_pages(path)
        ]
    return [
        page
        for path in sorted((ROOT / "data" / "demo_policies").glob("*.txt"))
        for page in extract_pages(path)
    ]


st.title("PolicyLens")
st.caption("Evidence-backed answers from company policies — with a separate reliability lab.")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader("Upload policy PDFs", type=["pdf"], accept_multiple_files=True)
    use_demo = st.toggle("Use included demo policies", value=not uploaded)
    chunking = st.selectbox("Chunking", ["Structure-aware", "Fixed-size"])
    embedding_options = ["Local hashing baseline"]
    if importlib.util.find_spec("sentence_transformers"):
        embedding_options.insert(0, "Sentence Transformer")
    if EmbeddingSettings.from_environment():
        embedding_options.append("API embeddings")
    embedder_name = st.selectbox("Embedding", embedding_options)
    method = st.selectbox(
        "Retrieval",
        ["semantic", "hybrid", "hybrid_rerank", "keyword"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    threshold = st.number_input("Refusal threshold", min_value=0.0, value=0.49, step=0.01, format="%.3f")
    generation_options = ["Extractive (offline)"]
    if importlib.util.find_spec("transformers"):
        generation_options.append("Local FLAN-T5 (experimental)")
    if LLMSettings.from_environment():
        generation_options.append("Configured LLM")
    generation_mode = st.selectbox("Answer generation", generation_options)

try:
    pages = []
    if use_demo:
        pages.extend(load_demo_pages())
    for file in uploaded:
        pages.extend(extract_uploaded_pdf(file.name, file.getvalue()))
    serialized = tuple((page.document, page.number, page.text) for page in pages)
    index = make_index(serialized, chunking, embedder_name) if pages else None
except (DocumentReadError, RuntimeError, ValueError) as exc:
    st.error(str(exc))
    index = None

ask_tab, lab_tab, dataset_tab, about_tab = st.tabs(
    ["Ask PolicyLens", "Evaluation Lab", "Dataset Builder", "How it works"]
)

with ask_tab:
    st.subheader("Ask a policy question")
    if index:
        st.caption(f"Indexed {len(index.chunks)} passages from {len(set(page.document for page in pages))} document(s).")
    question = st.text_input("Question", placeholder="How many unused leave days can be carried forward?")
    if st.button("Find answer", type="primary", disabled=not index or not question.strip()):
        generator = None
        if generation_mode == "Configured LLM":
            settings = LLMSettings.from_environment()
            generator = OpenAICompatibleGenerator(settings) if settings else None
        elif generation_mode == "Local FLAN-T5 (experimental)":
            generator = local_generator()
        try:
            answer = answer_question(index, question.strip(), method=method, threshold=threshold, generator=generator)
        except Exception as exc:
            st.error(f"The configured answer provider failed: {exc}")
            st.stop()
        if answer.status == "refused":
            st.warning(answer.text)
        else:
            st.success(answer.text)
        st.caption(
            f"Evidence score: {answer.confidence_score:.3f} · {answer.latency_ms:.1f} ms · "
            f"Generator: {answer.generator} · Estimated cost: ${answer.estimated_cost_usd:.6f}"
        )
        with st.expander("View sources", expanded=True):
            for item in answer.evidence:
                pages_label = ", ".join(str(page) for page in item.chunk.pages)
                st.markdown(f"**{item.rank}. {item.chunk.document} — page {pages_label}**")
                st.write(item.chunk.text)
                st.caption(f"Retrieval score: {item.score:.4f}")

with lab_tab:
    st.subheader("Evaluation Lab")
    st.info("This page is for developers and recruiters. Employees do not need it for normal Q&A.")
    case_path = ROOT / "data" / "benchmark_cases.csv"
    if not case_path.exists():
        case_path = ROOT / "data" / "evaluation_cases.csv"
    if index and case_path.exists() and st.button("Run demo evaluation"):
        cases = load_cases(case_path)
        with st.spinner("Running identical questions across all retrieval configurations..."):
            frames = []
            for name in ("semantic", "hybrid", "hybrid_rerank"):
                calibrated = calibrate_threshold(index, cases, name)
                frame = evaluate(index, cases, name, calibrated)
                frame["threshold"] = calibrated
                frames.append(frame)
            results = pd.concat(frames, ignore_index=True)
        summary = summarize(results)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        chart_data = summary[summary["split"] == "test"]
        if not chart_data.empty:
            st.plotly_chart(
                px.bar(chart_data, x="configuration", y=["hit@1", "hit@3", "mrr"], barmode="group"),
                use_container_width=True,
            )
        curve = threshold_sweep(index, [case for case in cases if case.split == "development"], method)
        selected_threshold = choose_threshold(curve)
        st.markdown("#### Development-set refusal trade-off")
        st.caption(f"Suggested demo threshold from development questions: {selected_threshold:.3f}")
        st.plotly_chart(
            px.line(curve, x="threshold", y=["correct_refusal_rate", "false_refusal_rate"], markers=True),
            use_container_width=True,
        )
        failures = add_failure_categories(results)
        failures = failures[failures["failure_category"] != "none"]
        st.markdown("#### Failure log")
        st.dataframe(
            failures[["configuration", "case_id", "question", "question_type", "failure_category", "correct_rank", "retrieved"]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button("Download per-query results", results.to_csv(index=False), "policylens_results.csv")
        st.download_button(
            "Download manual answer-grading sheet",
            build_manual_grading_rows(results).to_csv(index=False),
            "policylens_manual_grading.csv",
        )

        st.markdown("#### Chunking ablation")
        chunk_results = compare_chunking(pages, cases, index.embedder)
        chunk_summary = (
            chunk_results.groupby(["chunking", "split"])
            .agg(
                questions=("case_id", "count"),
                hit_at_1=("hit_at_1", "mean"),
                hit_at_3=("hit_at_3", "mean"),
                mrr=("reciprocal_rank", "mean"),
                latency_ms=("latency_ms", "mean"),
            )
            .reset_index()
        )
        st.dataframe(chunk_summary, use_container_width=True, hide_index=True)

with dataset_tab:
    st.subheader("Build the labelled evaluation set")
    st.write(
        "Create questions only from documents you have permission to use. Drafting can be assisted, "
        "but every expected answer and evidence page must be checked by a person."
    )
    template = pd.DataFrame(
        [
            {
                "id": "dev_001",
                "split": "development",
                "question": "How many leave days can be carried forward?",
                "expected_answer": "Five days",
                "relevant_documents": "leave_policy.pdf",
                "relevant_pages": "8|9",
                "question_type": "paraphrased",
                "answerable": "true",
            },
            {
                "id": "test_001",
                "split": "test",
                "question": "Does the policy include gym membership?",
                "expected_answer": "Not found",
                "relevant_documents": "",
                "relevant_pages": "",
                "question_type": "unanswerable",
                "answerable": "false",
            },
        ]
    )
    st.download_button("Download CSV template", template.to_csv(index=False), "policylens_cases_template.csv")
    candidate_file = st.file_uploader("Validate a question CSV", type=["csv"], key="evaluation_csv")
    if candidate_file:
        candidate = pd.read_csv(candidate_file, dtype=str).fillna("")
        required = set(template.columns)
        missing = sorted(required - set(candidate.columns))
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            development_count = int((candidate["split"].str.lower() == "development").sum())
            test_count = int((candidate["split"].str.lower() == "test").sum())
            answerable_count = int((candidate["answerable"].str.lower() == "true").sum())
            unanswerable_count = len(candidate) - answerable_count
            columns = st.columns(4)
            columns[0].metric("Questions", len(candidate))
            columns[1].metric("Development", development_count)
            columns[2].metric("Held-out test", test_count)
            columns[3].metric("Unanswerable", unanswerable_count)
            duplicate_ids = candidate["id"].duplicated().sum()
            unlabeled = candidate[
                (candidate["answerable"].str.lower() == "true")
                & ((candidate["relevant_documents"] == "") | (candidate["relevant_pages"] == ""))
            ]
            if duplicate_ids:
                st.error(f"Found {duplicate_ids} duplicate case ID(s).")
            if not unlabeled.empty:
                st.error(f"Found {len(unlabeled)} answerable case(s) without document/page evidence.")
            if not duplicate_ids and unlabeled.empty and development_count and test_count:
                st.success("The CSV has the required structure and basic labels.")
            st.dataframe(candidate.head(20), use_container_width=True, hide_index=True)

with about_tab:
    st.subheader("The pipeline")
    st.code(
        "PDF → extract text/pages → chunk → embed/index → retrieve → rerank\n"
        "→ evidence threshold → answer or refuse → cite sources",
        language="text",
    )
    st.markdown(
        "The included hashing embedder is an offline baseline. For a credible final benchmark, "
        "install the semantic extra and select the Sentence Transformer model. The app labels "
        "the baseline explicitly so its results cannot be mistaken for production semantic search."
    )
