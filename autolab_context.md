Maintained by hand (the generator reads it at each `propose`; it is part of the harness archive that deploy.sh ships). Update it when the competition rules or the findings change.

Competition conditions (organizers' call, 2026-11-09):
- All tasks are in English only; the tokenizer/length statistics may be taken from English text.
- Output beyond the budget is simply truncated, no hard limit on output is needed (a hard stop of an answer in progress is NOT a requirement).
- The budget (100 000 tokens, cost = input/10 + output) will most likely NOT be enough for all 100 tasks (coding, math, search, data_analysis). The key metric is therefore POINTS PER TOKEN and behaviour at budgets of 20 000-40 000, not "solve everything".
- Verification and rechecks are secondary: do not propose them unless they save tokens at the same points.
- The LLM is reached only through the proxy, four Qwen models: qwen3.6-35b-a3b(-noreason), qwen3.8-27b(-noreason).

Measured facts:
- Noise: one sd of points between repeats of an unchanged harness is 7.69 on mixed100 (3.4% of ~228 points at budget 100k). Acceptance needs paired 95% interval above 0 AND a difference of at least that; a smaller effect is `clarify` by definition.
- baseline0 at 100k spends only 56-66k (does not bind the budget); at 30k the budget binds and the order of tasks decides the points.
- Offline simulator (replays the real Scheduler on recorded outcomes) at budgets 20k/30k/40k: greedy 152/189/207 points, reward-first 150/172/192, fifo 131/158/167. It overestimates points by ~10% at a binding budget, the bias is shared by the arms.
- The predictor is weak: median relative error of the predicted task cost about 0.6 (coding 0.9, data_analysis 0.6, math 0.6, search 0.4-0.5); the greedy sorter ranks tasks by reward x p(success) / predicted tokens, so a better predictor can matter at a binding budget.
- Batching 8 math tasks with prompt `compute` saved 33-39% tokens at equal accuracy (one repeat, points +1 [-7, +11]: clarify).
- Rechecking with another model (`other`): on verify_frozen40 +9.0 points [+0.5, +20] (accept at the boundary), but on the fresh set verify_fresh40 with 3 repeats -0.67 [-6.7, +4.7] with +87% tokens: the effect was NOT confirmed. Verification costs 40-90% more tokens, which is a bad trade when the budget binds.
- The judge on search is useless (AUC 0.5-0.6); after the change of defaults it is off on search.
- The gateway's weekly and 3-hour limits count REQUESTS (1 000 000 per week), not tokens; tokens matter for the competition metric only.

Open questions (candidates for hypotheses):
- Does a different task order (reward-first, cheapest-first, type-aware, predictor `measured`/`flat`) beat greedy at budgets 20-40k on the real harness, not only in the simulator?
- Which cheap mode gives the most points per token: noreason models on all types or only on math/search, batching of math (4/8) with `compute`, fallback `late` vs `on` vs `off` for coding, censor on tokens?
- Is the unsolvable-task skip (HARNESS_PUNSOLV=contradiction, HARNESS_SUBMIT_UNSOLVABLE, HARNESS_GUESS_UNSOLVABLE) worth its tokens at a tight budget?
- Which types are worth their tokens at all when the budget binds (points per token by type)?
