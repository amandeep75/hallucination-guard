import functools
import inspect
import itertools
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import random
import time

TextLike = Union[str, Dict[str, Any]]

def _extract_text(x: TextLike) -> str:
    """
    Normalize function return values to a single text string to score.
    If dict, we expect the LLM output to be under common keys like 'answer', 'text', 'output'.
    You can customize this.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("answer", "text", "output", "content", "message"):
            if k in x and isinstance(x[k], str):
                return x[k]
        # fallback: dump json
        return json.dumps(x, ensure_ascii=False)
    return str(x)

def _similarity_score(texts: List[str]) -> float:
    """
    SelfCheck-style consistency: average pairwise cosine similarity.
    Using TF-IDF keeps it dependency-light and fast.
    Returns a value in [0, 1].
    """
    if len(texts) <= 1:
        return 1.0
    vect = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=5000)
    X = vect.fit_transform(texts)
    sim_matrix = cosine_similarity(X)
    # Upper triangle average, excluding diagonal
    n = len(texts)
    idxs = list(itertools.combinations(range(n), 2))
    if not idxs:
        return 1.0
    vals = [sim_matrix[i,j] for i,j in idxs]
    return float(sum(vals) / len(vals))

def _hallucination_score_from_consistency(consistency: float) -> float:
    """
    Map consistency → hallucination risk.
    Lower consistency => higher hallucination.
    """
    return max(0.0, min(1.0, 1.0 - consistency))

def _ensure_dict_return(val: TextLike) -> Dict[str, Any]:
    """
    Ensure the returned value is a dict so we can tag metadata without breaking callers.
    If the target function returned a string, we wrap it in {'answer': str}.
    If it returned a dict, we pass it through.
    """
    if isinstance(val, dict):
        return dict(val)  # shallow copy
    return {"answer": str(val)}

def _default_sampler(
    fn: Callable, n_samples: int, args: Tuple, kwargs: Dict
) -> List[TextLike]:
    """
    Call the wrapped function multiple times to induce diverse samples.
    We try to tweak kwargs if common generation knobs are present.
    This stays framework-agnostic (works with OpenAI, Ollama, HF, etc. if your fn forwards kwargs).
    """
    samples: List[TextLike] = []
    for i in range(n_samples):
        # best-effort to diversify
        k = dict(kwargs)
        # common knobs
        if "temperature" in k and isinstance(k["temperature"], (int, float)):
            k["temperature"] = max(0.2, min(1.3, float(k["temperature"]) + random.uniform(-0.2, 0.5)))
        elif "temperature" not in k:
            k["temperature"] = 0.9
        if "top_p" in k and isinstance(k["top_p"], (int, float)):
            k["top_p"] = max(0.5, min(1.0, float(k["top_p"]) + random.uniform(-0.2, 0.2)))
        if "seed" in k and isinstance(k["seed"], int):
            k["seed"] = k["seed"] + i + int(time.time()) % 1000
        try:
            out = fn(*args, **k)
            samples.append(out)
        except Exception as e:
            samples.append(f"[sample_error] {e}")
    return samples

def hallucination_guard(
    *,
    n_samples: int = 5,
    threshold: float = 0.40,
    action: str = "tag",   # "tag" | "warn" | "block"
    sampler: Optional[Callable[[Callable, int, Tuple, Dict], List[TextLike]]] = None,
    text_getter: Optional[Callable[[TextLike], str]] = None,
    attach_key: str = "hallucination_meta",
) -> Callable:
    """
    Decorator factory implementing a SelfCheck-style consistency check.

    - n_samples: number of generations to compare (>=3 recommended).
    - threshold: hallucination_score above this triggers the action.
    - action:
        "tag"  -> Include score + flag in response (non-intrusive).
        "warn" -> Include tag + prepend a warning string to the text.
        "block"-> Replace answer with a safe fallback message.
    - sampler: a function that returns a list of n_samples outputs for the same input.
               Default calls the wrapped function multiple times with tweaked gen knobs.
    - text_getter: optional custom extractor to pull text from complex return types.
    - attach_key: where to store meta in the returned dict.
    """
    if sampler is None:
        sampler = _default_sampler
    if text_getter is None:
        text_getter = _extract_text

    def decorator(fn: Callable) -> Callable:
        is_async = inspect.iscoroutinefunction(fn)

        async def _async_wrapper(*args, **kwargs):
            primary = await fn(*args, **kwargs)
            # Generate samples (call sync sampler in thread if needed)
            loop = asyncio.get_event_loop()
            samples = await loop.run_in_executor(None, sampler, fn, n_samples, args, kwargs)
            texts = [text_getter(s) for s in samples]
            consistency = _similarity_score(texts)
            hscore = _hallucination_score_from_consistency(consistency)

            out = _ensure_dict_return(primary)
            meta = {
                "n_samples": n_samples,
                "consistency_score": consistency,
                "hallucination_score": hscore,
                "threshold": threshold,
                "tripped": hscore >= threshold,
            }

            if hscore >= threshold:
                if action == "block":
                    out = {
                        "answer": "I'm not fully confident in this answer. Retrying or adding citations/context may help.",
                    }
                elif action == "warn":
                    # try to prepend a warning if the string is present
                    if isinstance(primary, str):
                        out["answer"] = f"Low confidence: {str(primary)}"
                    elif "answer" in out and isinstance(out["answer"], str):
                        out["answer"] = "Low confidence: " + out["answer"]
                    out["hallucination_warning"] = True
                else:
                    out["hallucination_warning"] = True
            else:
                out["hallucination_warning"] = False

            out[attach_key] = meta
            return out

        def _sync_wrapper(*args, **kwargs):
            primary = fn(*args, **kwargs)
            samples = sampler(fn, n_samples, args, kwargs)
            texts = [text_getter(s) for s in samples]
            consistency = _similarity_score(texts)
            hscore = _hallucination_score_from_consistency(consistency)

            out = _ensure_dict_return(primary)
            meta = {
                "n_samples": n_samples,
                "consistency_score": consistency,
                "hallucination_score": hscore,
                "threshold": threshold,
                "tripped": hscore >= threshold,
            }

            if hscore >= threshold:
                if action == "block":
                    out = {
                        "answer": "I’m not fully confident in this answer. Retrying or adding citations/context may help.",
                    }
                elif action == "warn":
                    if isinstance(primary, str):
                        out["answer"] = f"⚠️ Low confidence: {str(primary)}"
                    elif "answer" in out and isinstance(out["answer"], str):
                        out["answer"] = "⚠️ Low confidence: " + out["answer"]
                    out["hallucination_warning"] = True
                else:
                    out["hallucination_warning"] = True
            else:
                out["hallucination_warning"] = False

            out[attach_key] = meta
            return out

        return _async_wrapper if is_async else _sync_wrapper

    return decorator
