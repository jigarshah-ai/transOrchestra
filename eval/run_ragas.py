"""Ragas evaluation script for TransOrchestra RAG pipeline."""

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Suppress noisy deprecation warnings from third-party libraries during eval.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_NO_ANSWER_SENTINEL = "I don't have enough information in my documents to answer this reliably."

# Ensure project root is on the path.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_eval_set(path: str = None) -> List[dict]:
    """Load the evaluation question set from JSON."""
    eval_path = path or str(Path(__file__).parent / "eval_set.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_docs_from_vectorstore(vectorstore) -> List:
    """Reconstruct Document list from a ChromaDB collection."""
    from langchain_core.documents import Document
    raw = vectorstore._collection.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def build_hybrid_pipeline() -> Tuple:
    """Return (retriever, chain, fallback_retriever, fallback_chain) for hybrid+reranking.

    The fallback pair uses hybrid-only (no reranker) so evaluation continues
    gracefully when the Cohere API key is missing or invalid.
    """
    from backend.rag.embeddings import get_embedding_model
    from backend.rag.pipeline import build_rag_chain
    from backend.rag.retriever import build_hybrid_retriever, build_reranking_retriever
    from backend.rag.vectorstore import load_vectorstore

    embedding_model = get_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    docs = _get_docs_from_vectorstore(vectorstore)

    hybrid = build_hybrid_retriever(vectorstore, docs)
    retriever = build_reranking_retriever(hybrid, docs)
    chain = build_rag_chain(retriever)

    # Always build a pure-hybrid fallback for when the reranker call fails.
    fallback_retriever = build_hybrid_retriever(vectorstore, docs)
    fallback_chain = build_rag_chain(fallback_retriever)

    return retriever, chain, fallback_retriever, fallback_chain


def build_vector_only_pipeline() -> Tuple:
    """Return (retriever, chain) using the simple vector-only strategy."""
    from backend.rag.embeddings import get_embedding_model
    from backend.rag.pipeline import build_rag_chain
    from backend.rag.retriever import build_vector_retriever
    from backend.rag.vectorstore import load_vectorstore

    embedding_model = get_embedding_model()
    vectorstore = load_vectorstore(embedding_model)
    retriever = build_vector_retriever(vectorstore)
    chain = build_rag_chain(retriever)
    return retriever, chain, None, None


def _run_tavily_fallback(question: str) -> Optional[List]:
    """Run a Tavily web search and return results as Document objects, or None on failure."""
    try:
        from langchain_core.documents import Document

        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from backend.config import TAVILY_API_KEY
        if not TAVILY_API_KEY:
            return None

        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=question, max_results=3)
        docs = []
        for r in results.get("results", []):
            docs.append(Document(
                page_content=r.get("content", ""),
                metadata={"source": f"[WEB] {r.get('title', 'Web')}", "page": r.get("url", "")},
            ))
        return docs if docs else None
    except Exception as exc:
        print(f"\n    [Tavily fallback failed: {exc}]", end=" ")
        return None


def _run_one_query(chain, retriever, question: str,
                   fallback_chain=None, fallback_retriever=None):
    """Run a single question through RAG; use Tavily web search if RAG has no answer.

    This mirrors the corrective-RAG behaviour of the main safety agent so that
    evaluation scores reflect the real system capability, not just the local
    vector store coverage.
    """
    from backend.rag.pipeline import run_query_with_docs, run_query_with_web_context

    answer = None
    ctx_texts = ["No context retrieved"]

    # --- 1. Try primary retriever ---
    try:
        result = run_query_with_docs(chain, retriever, question)
        answer = result["answer"]
        try:
            raw_docs = retriever.invoke(question)
            ctx_texts = [d.page_content for d in raw_docs] or ctx_texts
        except Exception:
            pass
    except Exception as exc:
        # Reranker / Cohere failure — retry with fallback if available.
        if fallback_chain and fallback_retriever and (
            "401" in str(exc) or "cohere" in str(exc).lower() or "rerank" in str(exc).lower()
        ):
            try:
                result = run_query_with_docs(fallback_chain, fallback_retriever, question)
                answer = result["answer"]
                try:
                    raw_docs = fallback_retriever.invoke(question)
                    ctx_texts = [d.page_content for d in raw_docs] or ctx_texts
                except Exception:
                    pass
            except Exception as fb_exc:
                answer = f"Error: {fb_exc}"
        else:
            answer = f"Error: {exc}"

    # --- 2. Corrective RAG: if RAG couldn't answer, try Tavily web search ---
    if answer is None or _NO_ANSWER_SENTINEL in str(answer) or str(answer).startswith("Error:"):
        ret = fallback_retriever or retriever
        web_docs = _run_tavily_fallback(question)
        if web_docs:
            try:
                result = run_query_with_web_context(ret, question, web_docs)
                answer = result["answer"]
                # Context for Ragas = web snippets + any local chunks
                web_texts = [d.page_content for d in web_docs]
                try:
                    local_docs = ret.invoke(question)
                    local_texts = [d.page_content for d in local_docs]
                except Exception:
                    local_texts = []
                ctx_texts = web_texts + local_texts
                print("🌐", end=" ")
            except Exception as exc2:
                print(f"\n    [Web-context query failed: {exc2}]", end=" ")

    return answer or "Unable to retrieve answer.", ctx_texts


def run_evaluation(retriever, chain, eval_set: List[dict], label: str,
                   fallback_retriever=None, fallback_chain=None) -> dict:
    """Run the eval set through the pipeline and collect Ragas-compatible inputs."""
    print(f"\n[{label}] Running {len(eval_set)} evaluation questions…")

    # Ragas 0.1.x requires these exact singular column names.
    question_col, answer_col, contexts_col, ground_truth_col = [], [], [], []

    for i, item in enumerate(eval_set, 1):
        q = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i:02d}/{len(eval_set)}] {q[:70]}…", end=" ", flush=True)

        answer, ctx_texts = _run_one_query(
            chain, retriever, q, fallback_chain, fallback_retriever
        )

        question_col.append(q)
        answer_col.append(answer)
        contexts_col.append(ctx_texts)
        ground_truth_col.append(gt)
        print("✓")

    return {
        "question": question_col,      # Ragas requires singular
        "answer": answer_col,          # Ragas requires singular
        "contexts": contexts_col,
        "ground_truth": ground_truth_col,  # Ragas requires singular
    }


def compute_ragas_scores(eval_data: dict, label: str) -> dict:
    """Run Ragas faithfulness + answer_relevancy evaluation."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        ds = Dataset.from_dict(eval_data)
        print(f"\n[{label}] Running Ragas evaluation…")
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        scores = {
            "faithfulness": round(float(result["faithfulness"]), 4),
            "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        }
        return scores
    except Exception as exc:
        print(f"  ⚠️  Ragas evaluation failed: {exc}")
        return {"faithfulness": None, "answer_relevancy": None}


def print_results_table(label: str, scores: dict) -> None:
    """Print a formatted results table to stdout."""
    print(f"\n{'='*50}")
    print(f" Ragas Evaluation — {label}")
    print(f"{'='*50}")
    print(f" {'Metric':<25} | {'Score':>6}")
    print(f" {'-'*25}-+-{'-'*6}")
    f_score = scores.get("faithfulness")
    a_score = scores.get("answer_relevancy")
    print(f" {'Faithfulness':<25} | {f_score if f_score is not None else 'N/A':>6}")
    print(f" {'Answer Relevancy':<25} | {a_score if a_score is not None else 'N/A':>6}")
    print(f"{'='*50}")


def main() -> None:
    """Entry point for the Ragas evaluation script."""
    parser = argparse.ArgumentParser(description="TransOrchestra Ragas Evaluation")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run vector-only strategy for side-by-side comparison.",
    )
    parser.add_argument(
        "--eval_set",
        default=None,
        help="Path to a custom eval_set.json file.",
    )
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    print(f"Loaded {len(eval_set)} evaluation questions.")

    timestamp = datetime.now(UTC).isoformat()
    all_results: dict = {"timestamp": timestamp, "strategies": {}}

    # Primary evaluation — hybrid + reranker (with hybrid fallback on Cohere errors).
    print("\nLoading Hybrid + Reranking pipeline…")
    try:
        retriever_h, chain_h, fb_ret_h, fb_chain_h = build_hybrid_pipeline()
        eval_data_h = run_evaluation(
            retriever_h, chain_h, eval_set, "Hybrid+Reranker",
            fallback_retriever=fb_ret_h, fallback_chain=fb_chain_h,
        )
        scores_h = compute_ragas_scores(eval_data_h, "Hybrid+Reranker")
        print_results_table("Hybrid + Reranker", scores_h)
        all_results["strategies"]["hybrid_reranker"] = {
            "scores": scores_h,
            "raw": eval_data_h,
        }
    except FileNotFoundError as exc:
        print(f"\n❌ {exc}")
        print("Run scripts/ingest.py first to build the vector store.")
        sys.exit(1)

    # Optional comparison — vector only.
    if args.compare:
        print("\nLoading Vector-Only pipeline for comparison…")
        try:
            retriever_v, chain_v, _, _ = build_vector_only_pipeline()
            eval_data_v = run_evaluation(retriever_v, chain_v, eval_set, "Vector-Only")
            scores_v = compute_ragas_scores(eval_data_v, "Vector-Only")
            print_results_table("Vector Only", scores_v)
            all_results["strategies"]["vector_only"] = {
                "scores": scores_v,
                "raw": eval_data_v,
            }

            # Side-by-side comparison table.
            print(f"\n{'='*65}")
            print(" Side-by-Side Comparison")
            print(f"{'='*65}")
            print(f" {'Metric':<25} | {'Vector Only':>12} | {'Hybrid+Reranker':>15}")
            print(f" {'-'*25}-+-{'-'*12}-+-{'-'*15}")
            for metric in ("faithfulness", "answer_relevancy"):
                v = scores_v.get(metric, "N/A")
                h = scores_h.get(metric, "N/A")
                print(f" {metric.replace('_', ' ').title():<25} | {str(v):>12} | {str(h):>15}")
            print(f"{'='*65}")
        except Exception as exc:
            print(f"  ⚠️  Comparison pipeline failed: {exc}")

    # Save results.
    results_path = Path(__file__).parent / "results.json"
    try:
        saveable: dict = {"timestamp": timestamp, "strategies": {}}
        for k, v in all_results["strategies"].items():
            saveable["strategies"][k] = {
                "scores": v["scores"],
                "answers": v["raw"]["answer"],
                "questions": v["raw"]["question"],
            }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(saveable, f, indent=2)
        print(f"\n💾 Results saved to {results_path}")
    except Exception as exc:
        print(f"  ⚠️  Could not save results: {exc}")


if __name__ == "__main__":
    main()
