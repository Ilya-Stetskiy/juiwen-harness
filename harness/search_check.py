"""Free (0 tokens) checks of a search answer against the documents, and the shortlist for an independent second path.

No model opinion is used here: the earlier experiments showed that an LLM judge on search is useless (AUC ~0.6) and that
rejecting answers on a model's opinion cost points (verify_other on verify_fresh40). Everything is a string check of the
answer and the question against the task files.

    source_features(question, answer, files) -> dict      signals about where the answer sits in the sources
    doubt(feats, ...) -> (bool, reasons)                   cheap rule for "worth a second path"
    shortlist(question, files, k) -> [(name, text)]        top-k files by question overlap (+ files linked by the best one)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

STOP = set("""a an the of in on at to for from by with and or is are was were be been being as that this these those it its
which who whom whose what when where why how did do does done has have had not no yes than then there their they he she his
her him them you your we our i me my about into over under after before between during also can could would should will
shall may might must one two name named called find tell give using use file files directory kb markdown""".split())
_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.I)


def norm(s: str) -> str:
    return " ".join(_WORD.findall((s or "").lower()))


def terms(question: str) -> set[str]:
    return {w for w in _WORD.findall((question or "").lower()) if w not in STOP and len(w) > 1}


def read_files(ws) -> list[tuple[str, str]]:
    out = []
    if ws is None:
        return out
    for p in sorted(Path(ws).rglob("*")):
        if p.is_file():
            try:
                out.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                pass
    return out


def _idf(qt: set[str], files_norm: list[str]) -> dict[str, float]:
    n = len(files_norm) or 1
    return {t: math.log(1 + n / (1 + sum(t in f.split() for f in files_norm))) for t in qt}


def file_scores(question: str, files: list[tuple[str, str]]) -> list[float]:
    qt = terms(question)
    fn = [norm(t) for _, t in files]
    idf = _idf(qt, fn)
    tot = sum(idf.values()) or 1.0
    scores = []
    for f in fn:
        words = set(f.split())
        scores.append(sum(w for t, w in idf.items() if t in words) / tot)
    return scores


def shortlist(question: str, files: list[tuple[str, str]], k: int = 4) -> list[tuple[str, str]]:
    """Independent retrieval for the second path: the k files that share most (idf-weighted) question terms, plus files that
    mention a capitalised phrase of the best one's title line (one hop for multi-hop questions)."""
    if len(files) <= k:
        return list(files)
    sc = file_scores(question, files)
    order = sorted(range(len(files)), key=lambda i: -sc[i])
    pick = order[:k]
    best_name = Path(files[order[0]][0]).stem.replace("_", " ").lower()
    for i in order[k:]:
        if len(pick) >= k + 2:
            break
        if best_name and best_name in files[i][1].lower():
            pick.append(i)
    return [files[i] for i in sorted(pick)]


def _type_match(question: str, answer: str) -> int | None:
    q = (question or "").lower()
    a = answer or ""
    if re.search(r"\b(what year|which year|in what year|when was .* (founded|born|established|created))\b", q) or q.startswith("when "):
        return int(bool(re.search(r"\b\d{4}\b", a)))
    if q.startswith("how many") or "how many" in q[:40]:
        return int(bool(re.search(r"\d", a)))
    if q.startswith("who "):
        return int(bool(re.search(r"[A-Z]", a)))
    return None


def source_features(question: str, answer: str | None, files: list[tuple[str, str]], window: int = 250) -> dict:
    f: dict = {"sc_n_files": len(files)}
    if answer is None or not answer.strip():
        f["sc_answered"] = 0
        return f
    f["sc_answered"] = 1
    qt = terms(question)
    an = norm(answer)
    at = set(an.split())
    fn = [norm(t) for _, t in files]
    f["sc_ans_found"] = int(bool(an) and any(an in x for x in fn))
    alltok = set(w for x in fn for w in x.split())
    f["sc_tokcov"] = round(sum(w in alltok for w in at) / len(at), 3) if at else 0.0
    f["sc_ans_in_q"] = int(bool(at) and at <= set(norm(question).split()))
    f["sc_type_match"] = _type_match(question, answer)
    occ_files, n_occ, best_ctx = 0, 0, 0.0
    if an:
        for x in fn:
            c = x.count(an)
            if c:
                occ_files += 1
                n_occ += c
                start = 0
                for _ in range(min(c, 20)):
                    i = x.find(an, start)
                    if i < 0:
                        break
                    w = set(x[max(0, i - window): i + len(an) + window].split())
                    if qt:
                        best_ctx = max(best_ctx, sum(t in w for t in qt) / len(qt))
                    start = i + 1
    f["sc_n_occ_files"] = occ_files
    f["sc_n_occ"] = n_occ
    f["sc_ctx_overlap"] = round(best_ctx, 3)
    if files:
        sc = file_scores(question, files)
        top = max(range(len(files)), key=lambda i: sc[i])
        f["sc_top_overlap"] = round(sc[top], 3)
        f["sc_in_top_file"] = int(bool(an) and an in fn[top])
        ranks = sorted(range(len(files)), key=lambda i: -sc[i])
        hit = [r for r, i in enumerate(ranks) if an and an in fn[i]]
        f["sc_best_rank"] = hit[0] if hit else -1
    return f


def doubt(feats: dict, ctx_min: float = 0.5) -> tuple[bool, list[str]]:
    """Cheap doubt rule (b): nothing found / answer not in the sources / its context does not carry the question."""
    why = []
    if not feats.get("sc_answered"):
        why.append("no_answer")
        return True, why
    if not feats.get("sc_ans_found"):
        why.append("not_in_sources")
    elif feats.get("sc_ctx_overlap", 0.0) < ctx_min:
        why.append("weak_context")
    if feats.get("sc_ans_in_q"):
        why.append("copies_question")
    if feats.get("sc_type_match") == 0:
        why.append("type_mismatch")
    return bool(why), why
