#!/usr/bin/env python3
"""Narrow specialised task sets (4 types x 6 subgenres = 24 sets) for the lab: SPECS -> experiments/sets/nrw_*.yaml.

A narrow set is an ordinary frozen set (same keys as experiments/sets/*.yaml) plus two keys of tools/lab.py:
  select: {type: {field: cond}}   subgenre filter on the unified row (meta.*, check, instruction, task_id), see lab.row_matches
  exclude_ids_file:               frozen id list (cost-model training tasks + tasks of every older set), see sets/narrow/README.md
and a `narrow:` block (subgenre, description, target_cells = "block/type", reserve) that the lab ignores and autolab shows.

usage (local, writes yaml files without the `frozen:` block; freeze them on the host with `lab.py sets freeze nrw_*`):
  python tools/make_narrow_sets.py write [--out experiments/sets]
  python tools/make_narrow_sets.py noise [--out experiments]       # one noise-floor experiment per set (baseline, 3 repeats, budget 100k)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXCLUDE_FILE = "sets/narrow/exclude_ids.txt"
REWARD_LOW, REWARD_HIGH = 1, 5
NL = chr(10)

AGG = r"\b(how many|total|sum|difference|average|combined|number of|in all|altogether)\b"
BIG = r"\((?:[4-9]\d|\d{2,}\d) rows\)"
# (type, key, n, sources, select, unanswerable, title, description, target_cells)
SPECS = [
    # ---------------- math
    ("math", "gsm_chain", 32, ["gsm8k"], {"meta.calc_steps": [4, 5, 6, 7, 8, 9]}, None,
     "Grade-school word problems with long arithmetic chains (4-9 steps)",
     "GSM8K problems whose reference solution has 4+ calculation steps: a long chain where one slip ruins the answer; cheap in tokens, accuracy decided by step-by-step reasoning and the check.",
     ["solvers/math", "verifier/math", "recheck/math", "predictor/math"]),
    ("math", "word_simple", 32, ["svamp", "asdiv"], None, None,
     "One-step word problems (SVAMP + ASDiv)",
     "Short single-operation word problems, almost always solved; tokens per task are tiny and uniform, so the set measures overheads: batching, order, tokenizer, fixed prompt cost.",
     ["batcher/math", "sorter/math", "tokenizer/math", "predictor/math"]),
    ("math", "algebra", 30, ["math_algebra", "math_prealgebra"], None, None,
     "Symbolic algebra and prealgebra (MATH, all levels)",
     "MATH algebra/prealgebra with a numeric answer: equations, simplification, arithmetic; code computation helps; answers need exact formatting.",
     ["solvers/math", "executor/math", "censor/math"]),
    ("math", "discrete", 30, ["math_number_theory", "math_counting_and_probability"], None, None,
     "Number theory, counting and probability (MATH)",
     "Discrete mathematics: divisibility, modular arithmetic, combinatorics, probability; brute force in python often decides, wrong answers are confident; probability tasks cost the most.",
     ["executor/math", "solvers/math", "guess/math"]),
    ("math", "geometry", 24, ["math_geometry"], None, None,
     "Geometry (MATH)",
     "MATH geometry with a numeric answer; the figure is only given as text/asy code, so many tasks are hard to solve and expensive (the most costly math source); tests giving up and the answer check.",
     ["solvers/math", "verifier/math", "guess/math", "budget/math"]),
    ("math", "hard_symbolic", 24, ["math_intermediate_algebra", "math_precalculus"], {"meta.level": ["Level 4", "Level 5"]}, None,
     "Hard symbolic mathematics (intermediate algebra and precalculus, levels 4-5)",
     "Competition-style hard problems: long outputs, high variance of cost, a sizeable share of unsolvable tasks; tests budget stop, cost censoring, loop detector, fallback.",
     ["budget/math", "censor/math", "fallback/math", "predictor/math", "guess/math"]),
    # ---------------- search
    ("search", "comparison", 32, ["hotpotqa", "2wikimultihopqa"], {"meta.type": ["comparison"]}, None,
     "Comparison questions (HotpotQA + 2Wiki)",
     "Two entities, one attribute, pick one or yes/no: two parallel lookups in kb/*.md, short answers; tests retrieval over 10 paragraphs and the answer check.",
     ["solvers/search", "executor/search", "verifier/search"]),
    ("search", "bridge", 32, ["hotpotqa", "2wikimultihopqa"], {"meta.type": ["bridge", "compositional"]}, None,
     "Bridge / compositional two-hop questions (HotpotQA + 2Wiki)",
     "The answer needs an intermediate entity found first (bridge in HotpotQA, compositional in 2Wiki); a chain of two lookups where the first mistake is silent; tests the search loop and rechecks.",
     ["solvers/search", "recheck/search", "executor/search"]),
    ("search", "multihop_deep", 30, ["musique_full"], {"meta.hops": [3, 4]}, 0,
     "Deep multi-hop questions, 3-4 hops (MuSiQue, answerable)",
     "MuSiQue questions that need 3-4 hops over 20 paragraphs, all answerable; the hardest and most expensive search tasks, accuracy low; tests budget stop and cost prediction.",
     ["budget/search", "censor/search", "predictor/search", "solvers/search"]),
    ("search", "multihop_abstain", 30, ["musique_full"], {"meta.hops": [2]}, 15,
     "Two-hop questions with unanswerable half (MuSiQue)",
     "Two-hop MuSiQue questions, 15 of 30 unanswerable (the answerable twin has the same question; the unanswerable one has a paragraph removed): tests recognising a missing answer, guessing and verifier recheck. NOTE the 50% share is deliberate (the competition has about 15-25%) so the abstention behaviour has power.",
     ["guess/search", "verifier/search", "recheck/search"]),
    ("search", "factoid", 32, ["squad_v2"], {"task_id": {"re": "[0-4]$"}}, 0,
     "Single-hop factoid questions (SQuAD v2, answerable)",
     "One paragraph, one span answer; very cheap (about 50 tokens per task), many requests; tests batching, sorting and the tokenizer, and accuracy of short exact-string answers.",
     ["batcher/search", "sorter/search", "tokenizer/search", "predictor/search"]),
    ("search", "factoid_abstain", 32, ["squad_v2"], {"task_id": {"re": "[5-9]$"}}, 16,
     "Single-hop questions with unanswerable half (SQuAD v2)",
     "SQuAD v2 with 16 of 32 unanswerable (plausible-sounding question whose answer is not in the paragraph): the cleanest test of abstention, guess and the free verifier signals. The 50% share is deliberate (competition: about 15-25%).",
     ["guess/search", "verifier/search", "fallback/search"]),
    # ---------------- data_analysis
    ("data_analysis", "tabfact", 32, ["tabfact"], None, None,
     "Table fact checking, true/false (TabFact)",
     "One small table and a statement; binary answer (chance level 50%); cheap, many tasks; tests the guess prior, batching and the verifier on a binary label.",
     ["batcher/data_analysis", "guess/data_analysis", "verifier/data_analysis", "tokenizer/data_analysis"]),
    ("data_analysis", "wtq_lookup", 30, ["wikitablequestions"], {"check": ["exact_str"], "instruction": {"not_re": AGG + "|" + BIG}}, None,
     "Table lookup questions on small tables (WikiTableQuestions)",
     "Single-value questions (who/which/when/where) on tables under 40 rows, no aggregation words: select the right row/column and copy the cell exactly.",
     ["solvers/data_analysis", "executor/data_analysis", "sandbox/data_analysis"]),
    ("data_analysis", "wtq_aggregate", 30, ["wikitablequestions"], {"check": ["exact_str"], "instruction": {"re": AGG, "not_re": BIG}}, None,
     "Table aggregation questions (WikiTableQuestions: count, total, difference, average)",
     "Questions with how many / total / difference / average on tables under 40 rows: a number computed with python from several rows; wrong filter or off-by-one is the typical error.",
     ["executor/data_analysis", "sandbox/data_analysis", "solvers/data_analysis", "verifier/data_analysis"]),
    ("data_analysis", "wtq_list", 24, ["wikitablequestions"], {"check": ["exact_set"], "instruction": {"not_re": BIG}}, None,
     "Table questions with a list answer (WikiTableQuestions, exact_set)",
     "The answer is a list of items separated by '|'; scoring is by set match, so a missing or extra item fails the task; tests answer formatting and the verifier on multi-part answers.",
     ["solvers/data_analysis", "verifier/data_analysis", "guess/data_analysis"]),
    ("data_analysis", "wtq_big_table", 24, ["wikitablequestions"], {"check": ["exact_str"], "instruction": {"re": BIG}}, None,
     "Questions on large tables, 40+ rows (WikiTableQuestions)",
     "Tables of 40 and more rows: long prompt/workspace, more tokens per task and more variance; tests cost prediction from workspace size, the order, the budget and the tokenizer.",
     ["predictor/data_analysis", "sorter/data_analysis", "budget/data_analysis", "tokenizer/data_analysis"]),
    ("data_analysis", "dabench", 30, ["dabench"], None, None,
     "Data analysis on csv files with a strict answer format (DABench)",
     "Statistics, correlation, outliers, feature engineering and ML questions on real csv files; answers in @name[value] format with rounding; accuracy lowest of the type (about 40%), costly; tests executor/sandbox and format handling. Only tasks outside the cost-model training are free, so the set is a single DABench set (no easy/hard split).",
     ["executor/data_analysis", "sandbox/data_analysis", "budget/data_analysis", "censor/data_analysis"]),
    # ---------------- coding
    ("coding", "mbpp", 32, ["mbpp"], None, None,
     "Implement a function from a description (MBPP)",
     "Write a function from a stub and description; unittest checks; short, often one shot; tests the sandbox run, sorter and the tokenizer; no bug to find.",
     ["solvers/coding", "sandbox/coding", "sorter/coding", "tokenizer/coding"]),
    ("coding", "cg_single_bug", 30, ["codegen_fix"], {"meta.n_bugs": [1], "meta.pool": ["main"], "meta.infeasible": ["False"]}, 0,
     "Repair a library with one bug (codegen_fix)",
     "mylib.py of 3-8 functions with exactly one injected bug, tests per function; the cheapest repair task; tests locating a single fault.",
     ["solvers/coding", "executor/coding", "sorter/coding", "predictor/coding"]),
    ("coding", "cg_multi_bug", 24, ["codegen_fix"], {"meta.n_bugs": [3, 4], "meta.pool": ["main"], "meta.infeasible": ["False"]}, 0,
     "Repair a library with 3-4 bugs (codegen_fix)",
     "Three or four injected bugs in one file: several fix-test rounds, long outputs, partial fixes; the most expensive feasible coding tasks; tests budget, censor and late fallback.",
     ["predictor/coding", "budget/coding", "censor/coding", "fallback/coding"]),
    ("coding", "cg_toy_lib", 30, ["codegen_fix"], {"meta.pool": ["toy"], "meta.infeasible": ["False"]}, 0,
     "Repair of toy libraries like the organisers' example (codegen_fix, toy pool)",
     "Tasks built from the 42 hand-written toy functions in the style of the organisers' coding_0003 (about 20% of codegen_fix); small, similar to the real competition examples; tests solver prompts tuned to that style.",
     ["solvers/coding", "executor/coding", "guard/coding"]),
    ("coding", "cg_large_file", 24, ["codegen_fix"], {"meta.n_bugs": [2], "meta.pool": ["main"], "meta.infeasible": ["False"], "meta.n_functions": [6, 7, 8]}, 0,
     "Repair of large files, 6-8 functions with 2 bugs (codegen_fix)",
     "Larger files with two bugs: long prompts and test output; tests cost prediction from workspace size and the order of tasks.",
     ["predictor/coding", "sorter/coding", "tokenizer/coding", "budget/coding"]),
    ("coding", "cg_infeasible", 30, ["codegen_fix"], {"meta.n_bugs": [2], "meta.pool": ["main"], "meta.n_functions": [3, 4, 5]}, 15,
     "Repair tasks with an impossible half (codegen_fix, contradictory test)",
     "15 of 30 tasks cannot be solved (a second test with the same input and another expected value); success is recognising that and not burning budget; the 50% share is deliberate (competition: about 15-25%). The feasible half has two bugs.",
     ["guess/coding", "verifier/coding", "budget/coding", "guard/coding"]),
]


def name(typ: str, key: str) -> str:
    return f"nrw_{typ}_{key}"


def seed_of(i: int) -> int:
    return 7100 + i


def j(x) -> str:
    return json.dumps(x, ensure_ascii=False)


def set_yaml(i: int, spec) -> str:
    typ, key, n, sources, select, unans, title, desc, cells = spec
    lines = [
        f"# Narrow set {name(typ, key)}: {title}",
        "# Frozen by `lab.py sets freeze` (hash below); built by tools/make_narrow_sets.py; description and task list in sets/narrow/README.md.",
        "role: dev",
        f"tasks: {{{typ}: {n}}}",
        f"sources: {{{typ}: {j(sources)}}}",
    ]
    if select:
        lines.append(f"select: {{{typ}: {j(select)}}}")
    if unans is not None:
        lines.append(f"unanswerable: {{{typ}: {unans}}}")
    lines += [
        f"rewards: {{low: {REWARD_LOW}, high: {REWARD_HIGH}, seed: {seed_of(i)}}}",
        f"seed: {seed_of(i)}",
        f"exclude_ids_file: {EXCLUDE_FILE}",
        "narrow:",
        f"  type: {typ}",
        f"  subgenre: {key}",
        f"  title: {j(title)}",
        f"  description: {j(desc)}",
        f"  target_cells: {j(cells)}",
        f"  unanswerable_share: {unans / n if unans is not None else 0}",
        f"  reserve_seed: {seed_of(i) + 1000}",
        "  reserve: sets/narrow/manifest.json",
    ]
    return "\n".join(lines) + "\n"


BIG_RUNS = {"coding_mbpp", "coding_cg_multi_bug", "coding_cg_large_file", "coding_cg_infeasible", "coding_cg_single_bug"}   # estimated 60k+ tokens per run: 100k could bind


def noise_yaml(typ: str, key: str, n: int, title: str) -> str:
    nm = name(typ, key)
    budget = 300000 if f"{typ}_{key}" in BIG_RUNS else 100000
    return f"""name: noise_{nm}
hypothesis: >
  Noise floor of the narrow set {nm} ({title}, {n} tasks): the unchanged baseline harness (the as_is arm of baseline0:
  model auto, fallback on) is run 3 times, so the standard deviation of points between
  repeats of an unchanged harness is known for this set. A difference below it is never accepted on this set.
  Budget {budget} does not bind (estimated tokens per run are far below it), so there is no truncation.
backend: gateway
model: auto
set: {nm}
repeats: 3
budget: {budget}
arms:
  as_is:
    env: {{HARNESS_FALLBACK: "on"}}
"""


def noise_local_yaml(typ: str, key: str, n: int, title: str) -> str:
    nm = name(typ, key)
    budget = 300000 if f"{typ}_{key}" in BIG_RUNS else 100000
    return f"""name: noise_local_{nm}
hypothesis: >
  PROVISIONAL noise floor of the narrow set {nm} ({title}, {n} tasks) on the LOCAL Qwen (llama-server :8011 behind the shim :18444,
  no gateway tokens): the same unchanged baseline as noise_{nm} (as_is: model auto, fallback on), 3 repeats, same budget.
  `model: auto` keeps the harness routing over the four model ids; the shim serves all of them with the one local Qwen3.6-35B-A3B Q4
  (a -noreason id = thinking off), so the sd is only an APPROXIMATION of the gateway sd (scaled by tools/narrow_noise.py after the
  gateway check on cheap sets).
backend: local
model: auto
workers: 3
set: {nm}
repeats: 3
budget: {budget}
arms:
  as_is:
    env: {{HARNESS_FALLBACK: "on"}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["write", "noise", "noise_local"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    root = HERE.parent / "experiments"
    out = Path(a.out) if a.out else (root / "sets" if a.action == "write" else root)
    if a.action == "write":
        for i, s in enumerate(SPECS):
            (out / f"{name(s[0], s[1])}.yaml").write_text(set_yaml(i, s), encoding="utf-8", newline=NL)
        print(f"{len(SPECS)} set files -> {out}")
    elif a.action == "noise_local":
        for s in SPECS:
            (out / f"noise_local_{name(s[0], s[1])}.yaml").write_text(noise_local_yaml(s[0], s[1], s[2], s[6]), encoding="utf-8", newline=NL)
        print(f"{len(SPECS)} local noise experiments -> {out}")
    else:
        for s in SPECS:
            (out / f"noise_{name(s[0], s[1])}.yaml").write_text(noise_yaml(s[0], s[1], s[2], s[6]), encoding="utf-8", newline=NL)
        print(f"{len(SPECS)} noise experiments -> {out}")


if __name__ == "__main__":
    main()
