# Узкие специализированные наборы (24 набора: 4 типа × 6 поджанров)

Дата: 2026-11-10. Зачем: гипотезу автолаборатории про блок схемы (`status/`) можно проверять не на общей смеси `mixed100_dev`, а на узких данных, которые нагружают именно этот блок и тип. Все наборы заморожены (хеш в `frozen:`), роль `dev` (метка kind), общий порог шума смеси (7.69) к ним НЕ применим: у каждого набора свой порог, замеры подготовлены (раздел «Шумовой замер»).

## Как устроено

- Файлы `experiments/sets/nrw_<тип>_<поджанр>.yaml` лежат рядом с остальными замороженными наборами (лаборатория ищет наборы только в `experiments/sets/*.yaml`), формат тот же: `role`, `tasks`, `sources`, `unanswerable`, `rewards` (случайные 1-5, свой seed), `seed`, `frozen`. Подключение: `set: nrw_math_geometry` в эксперименте.
- Новые ключи (поддержаны в `tools/lab.py`, остальные наборы не затронуты): `select: {тип: {поле: условие}}` (фильтр поджанра по полям записи: `meta.*`, `check`, `instruction`, `task_id`; условие: список значений или `{re: ..}` / `{not_re: ..}`) и `exclude_ids_file` (замороженный список исключаемых id).
- Блок `narrow:` (название, описание, `target_cells` = список `блок/тип` из схемы статусов, `reserve_seed`) лаборатория не читает; его показывает автолаборатория (`tools/autolab.py`: `check_sets()` и `sets_available()` добавляют `narrow <поджанр>, loads cells: ...`; в кнопках команды `проверка <имя>` набор, чьи ячейки совпадают с целевыми ячейками гипотезы, идёт первым).
- `manifest.json`: для каждого набора список задач, резерв (`reserve`), ячейки, оценки стоимости, хеш. `all_task_ids.txt`: все id наборов и резервов (**добавьте их в исключения при переобучении gbm**, см. ниже).
- Поджанры выбраны по тому, какие блоки они нагружают (решатели, батчер, предиктор, верификатор, guess, бюджет, ...), а не по источнику; на источнике держатся только те, где у данных нет метки. В `~/dataset-full/unified` нет меток «строки/алгоритмы» для coding и «многотабличных» задач, подтипы агрегаций выделены по тексту вопроса.

## Таблица наборов

Стоимость одного прогона (один повтор, одна рука, модель `auto`): токены по лабораторной формуле (вход/10 + выход); в скобках верхняя оценка по потоку (старый харнесс, 4 модели, без калибровки); запросы шлюза = вызовы модели × 2.5 (по лаборатории ~250 запросов на 100 задач при ~1 вызове на задачу); минуты по лимиту 55 запросов в минуту и 5 воркерам. Точность оценок порядка ×2; у `nrw_coding_mbpp` поток даёт 7.8 вызова и 213 с на задачу, это выброс: цифру уточнит первый же запуск. «Нерешаемых» только там, где в данных есть честная метка (`check: unanswerable`); в `*_abstain` и `cg_infeasible` их доля 50% нарочно (в соревновании около 15-25%), иначе у проверки отказа нет мощности; в остальных наборах нерешаемых нет, но часть задач решается плохо (см. долю верных).

| тип | набор | задач | нерешаемых | источник (задач) | нагружаемые ячейки (блок/тип) | токены/прогон | запросы шлюза | минуты |
|---|---|---|---|---|---|---|---|---|
| math | `nrw_math_algebra` | 30 | 0 | math_algebra 16, math_prealgebra 14 | solvers/math, executor/math, censor/math | 18780 (16121) | 68 | 4.0 |
| math | `nrw_math_discrete` | 30 | 0 | math_counting_and_probability 16, math_number_theory 14 | executor/math, solvers/math, guess/math | 26700 (23940) | 98 | 4.0 |
| math | `nrw_math_geometry` | 24 | 0 | math_geometry 24 | solvers/math, verifier/math, guess/math, budget/math | 32232 (43807) | 105 | 6.3 |
| math | `nrw_math_gsm_chain` | 32 | 0 | gsm8k 32 | solvers/math, verifier/math, recheck/math, predictor/math | 6624 (13038) | 50 | 4.1 |
| math | `nrw_math_hard_symbolic` | 24 | 0 | math_precalculus 12, math_intermediate_algebra 12 | budget/math, censor/math, fallback/math, predictor/math, guess/math | 35736 (48926) | 100 | 5.6 |
| math | `nrw_math_word_simple` | 32 | 0 | asdiv 16, svamp 16 | batcher/math, sorter/math, tokenizer/math, predictor/math | 9408 (8446) | 68 | 3.4 |
| search | `nrw_search_bridge` | 32 | 0 | hotpotqa 17, 2wikimultihopqa 15 | solvers/search, recheck/search, executor/search | 5632 (9829) | 88 | 3.4 |
| search | `nrw_search_comparison` | 32 | 0 | 2wikimultihopqa 21, hotpotqa 11 | solvers/search, executor/search, verifier/search | 5312 (9271) | 88 | 3.3 |
| search | `nrw_search_factoid` | 32 | 0 | squad_v2 32 | batcher/search, sorter/search, tokenizer/search, predictor/search | 1568 (6827) | 82 | 3.4 |
| search | `nrw_search_factoid_abstain` | 32 | 16 | squad_v2 32 | guess/search, verifier/search, fallback/search | 1568 (6835) | 85 | 3.3 |
| search | `nrw_search_multihop_abstain` | 30 | 15 | musique_full 30 | guess/search, verifier/search, recheck/search | 9810 (15993) | 80 | 3.3 |
| search | `nrw_search_multihop_deep` | 30 | 0 | musique_full 30 | budget/search, censor/search, predictor/search, solvers/search | 11850 (19315) | 82 | 3.6 |
| data_analysis | `nrw_data_analysis_dabench` | 30 | 0 | dabench 30 | executor/data_analysis, sandbox/data_analysis, budget/data_analysis, censor/data_analysis | 19710 (41940) | 75 | 6.9 |
| data_analysis | `nrw_data_analysis_tabfact` | 32 | 0 | tabfact 32 | batcher/data_analysis, guess/data_analysis, verifier/data_analysis, tokenizer/data_analysis | 13472 (24844) | 110 | 4.6 |
| data_analysis | `nrw_data_analysis_wtq_aggregate` | 30 | 0 | wikitablequestions 30 | executor/data_analysis, sandbox/data_analysis, solvers/data_analysis, verifier/data_analysis | 9660 (19112) | 80 | 4.1 |
| data_analysis | `nrw_data_analysis_wtq_big_table` | 24 | 0 | wikitablequestions 24 | predictor/data_analysis, sorter/data_analysis, budget/data_analysis, tokenizer/data_analysis | 9624 (19018) | 72 | 3.5 |
| data_analysis | `nrw_data_analysis_wtq_list` | 24 | 0 | wikitablequestions 24 | solvers/data_analysis, verifier/data_analysis, guess/data_analysis | 9048 (17881) | 68 | 3.4 |
| data_analysis | `nrw_data_analysis_wtq_lookup` | 30 | 0 | wikitablequestions 30 | solvers/data_analysis, executor/data_analysis, sandbox/data_analysis | 12090 (23924) | 92 | 4.7 |
| coding | `nrw_coding_cg_infeasible` | 30 | 15 | codegen_fix 30 | guess/coding, verifier/coding, budget/coding, guard/coding | 31020 (81985) | 100 | 6.5 |
| coding | `nrw_coding_cg_large_file` | 24 | 0 | codegen_fix 24 | predictor/coding, sorter/coding, tokenizer/coding, budget/coding | 28200 (74512) | 85 | 5.2 |
| coding | `nrw_coding_cg_multi_bug` | 24 | 0 | codegen_fix 24 | predictor/coding, budget/coding, censor/coding, fallback/coding | 37704 (99674) | 95 | 6.5 |
| coding | `nrw_coding_cg_single_bug` | 30 | 0 | codegen_fix 30 | solvers/coding, executor/coding, sorter/coding, predictor/coding | 23520 (62167) | 85 | 5.4 |
| coding | `nrw_coding_cg_toy_lib` | 30 | 0 | codegen_fix 30 | solvers/coding, executor/coding, guard/coding | 5400 (14307) | 38 | 3.2 |
| coding | `nrw_coding_mbpp` | 32 | 0 | mbpp 32 | solvers/coding, sandbox/coding, sorter/coding, tokenizer/coding | 89856 (124492) | 522 | 22.7 |
| **итого** | 24 набора | 700 | | | | 454524 (826204) | 2416 | 124 |

## Описания

- **`nrw_math_algebra`** (30 задач, хеш `237f5e537f05`): Symbolic algebra and prealgebra (MATH, all levels). MATH algebra/prealgebra with a numeric answer: equations, simplification, arithmetic; code computation helps; answers need exact formatting. Доля верных в потоке для пула такого вида: 0.9.
- **`nrw_math_discrete`** (30 задач, хеш `417ea1a63169`): Number theory, counting and probability (MATH). Discrete mathematics: divisibility, modular arithmetic, combinatorics, probability; brute force in python often decides, wrong answers are confident; probability tasks cost the most. Доля верных в потоке для пула такого вида: 0.91.
- **`nrw_math_geometry`** (24 задач, хеш `9fef31bf2103`): Geometry (MATH). MATH geometry with a numeric answer; the figure is only given as text/asy code, so many tasks are hard to solve and expensive (the most costly math source); tests giving up and the answer check. Доля верных в потоке для пула такого вида: 0.73.
- **`nrw_math_gsm_chain`** (32 задач, хеш `b256cf1cedc3`): Grade-school word problems with long arithmetic chains (4-9 steps). GSM8K problems whose reference solution has 4+ calculation steps: a long chain where one slip ruins the answer; cheap in tokens, accuracy decided by step-by-step reasoning and the check. Доля верных в потоке для пула такого вида: 0.91.
- **`nrw_math_hard_symbolic`** (24 задач, хеш `3ffa51f4b3e2`): Hard symbolic mathematics (intermediate algebra and precalculus, levels 4-5). Competition-style hard problems: long outputs, high variance of cost, a sizeable share of unsolvable tasks; tests budget stop, cost censoring, loop detector, fallback. Доля верных в потоке для пула такого вида: 0.78.
- **`nrw_math_word_simple`** (32 задач, хеш `8731791e83af`): One-step word problems (SVAMP + ASDiv). Short single-operation word problems, almost always solved; tokens per task are tiny and uniform, so the set measures overheads: batching, order, tokenizer, fixed prompt cost. Доля верных в потоке для пула такого вида: 0.96.
- **`nrw_search_bridge`** (32 задач, хеш `bc32d06fde4c`): Bridge / compositional two-hop questions (HotpotQA + 2Wiki). The answer needs an intermediate entity found first (bridge in HotpotQA, compositional in 2Wiki); a chain of two lookups where the first mistake is silent; tests the search loop and rechecks. Доля верных в потоке для пула такого вида: 0.71.
- **`nrw_search_comparison`** (32 задач, хеш `47a7ce07e126`): Comparison questions (HotpotQA + 2Wiki). Two entities, one attribute, pick one or yes/no: two parallel lookups in kb/*.md, short answers; tests retrieval over 10 paragraphs and the answer check. Доля верных в потоке для пула такого вида: 0.87.
- **`nrw_search_factoid`** (32 задач, хеш `0c6dc09b413b`): Single-hop factoid questions (SQuAD v2, answerable). One paragraph, one span answer; very cheap (about 50 tokens per task), many requests; tests batching, sorting and the tokenizer, and accuracy of short exact-string answers. Доля верных в потоке для пула такого вида: 0.68.
- **`nrw_search_factoid_abstain`** (32 задач, хеш `394ea4a474d1`): Single-hop questions with unanswerable half (SQuAD v2). SQuAD v2 with 16 of 32 unanswerable (plausible-sounding question whose answer is not in the paragraph): the cleanest test of abstention, guess and the free verifier signals. The 50% share is deliberate (competition: about 15-25%). Доля верных в потоке для пула такого вида: 0.64.
- **`nrw_search_multihop_abstain`** (30 задач, хеш `54c773a6ef94`): Two-hop questions with unanswerable half (MuSiQue). Two-hop MuSiQue questions, 15 of 30 unanswerable (the answerable twin has the same question; the unanswerable one has a paragraph removed): tests recognising a missing answer, guessing and verifier recheck. NOTE the 50% share is deliberate (the competition has about 15-25%) so the abstention behaviour has power. Доля верных в потоке для пула такого вида: 0.71.
- **`nrw_search_multihop_deep`** (30 задач, хеш `c176a0eafff8`): Deep multi-hop questions, 3-4 hops (MuSiQue, answerable). MuSiQue questions that need 3-4 hops over 20 paragraphs, all answerable; the hardest and most expensive search tasks, accuracy low; tests budget stop and cost prediction. Доля верных в потоке для пула такого вида: 0.5.
- **`nrw_data_analysis_dabench`** (30 задач, хеш `4e75c1c1f6ad`): Data analysis on csv files with a strict answer format (DABench). Statistics, correlation, outliers, feature engineering and ML questions on real csv files; answers in @name[value] format with rounding; accuracy lowest of the type (about 40%), costly; tests executor/sandbox and format handling. Only tasks outside the cost-model training are free, so the set is a single DABench set (no easy/hard split). Доля верных в потоке для пула такого вида: 0.36.
- **`nrw_data_analysis_tabfact`** (32 задач, хеш `a4358d63b43f`): Table fact checking, true/false (TabFact). One small table and a statement; binary answer (chance level 50%); cheap, many tasks; tests the guess prior, batching and the verifier on a binary label. Доля верных в потоке для пула такого вида: 0.76.
- **`nrw_data_analysis_wtq_aggregate`** (30 задач, хеш `72fc8dfb9781`): Table aggregation questions (WikiTableQuestions: count, total, difference, average). Questions with how many / total / difference / average on tables under 40 rows: a number computed with python from several rows; wrong filter or off-by-one is the typical error. Доля верных в потоке для пула такого вида: 0.72.
- **`nrw_data_analysis_wtq_big_table`** (24 задач, хеш `f69dac940e57`): Questions on large tables, 40+ rows (WikiTableQuestions). Tables of 40 and more rows: long prompt/workspace, more tokens per task and more variance; tests cost prediction from workspace size, the order, the budget and the tokenizer. Доля верных в потоке для пула такого вида: 0.68.
- **`nrw_data_analysis_wtq_list`** (24 задач, хеш `6469d123dac3`): Table questions with a list answer (WikiTableQuestions, exact_set). The answer is a list of items separated by '|'; scoring is by set match, so a missing or extra item fails the task; tests answer formatting and the verifier on multi-part answers. Доля верных в потоке для пула такого вида: 0.77.
- **`nrw_data_analysis_wtq_lookup`** (30 задач, хеш `b877d2dfe866`): Table lookup questions on small tables (WikiTableQuestions). Single-value questions (who/which/when/where) on tables under 40 rows, no aggregation words: select the right row/column and copy the cell exactly. Доля верных в потоке для пула такого вида: 0.71.
- **`nrw_coding_cg_infeasible`** (30 задач, хеш `223520ee47f0`): Repair tasks with an impossible half (codegen_fix, contradictory test). 15 of 30 tasks cannot be solved (a second test with the same input and another expected value); success is recognising that and not burning budget; the 50% share is deliberate (competition: about 15-25%). The feasible half has two bugs. Доля верных в потоке для пула такого вида: 0.95.
- **`nrw_coding_cg_large_file`** (24 задач, хеш `36c398cc076f`): Repair of large files, 6-8 functions with 2 bugs (codegen_fix). Larger files with two bugs: long prompts and test output; tests cost prediction from workspace size and the order of tasks. Доля верных в потоке для пула такого вида: 0.95.
- **`nrw_coding_cg_multi_bug`** (24 задач, хеш `d332b65d515b`): Repair a library with 3-4 bugs (codegen_fix). Three or four injected bugs in one file: several fix-test rounds, long outputs, partial fixes; the most expensive feasible coding tasks; tests budget, censor and late fallback. Доля верных в потоке для пула такого вида: 0.92.
- **`nrw_coding_cg_single_bug`** (30 задач, хеш `dd545bc9f911`): Repair a library with one bug (codegen_fix). mylib.py of 3-8 functions with exactly one injected bug, tests per function; the cheapest repair task; tests locating a single fault. Доля верных в потоке для пула такого вида: 0.97.
- **`nrw_coding_cg_toy_lib`** (30 задач, хеш `842399eb826a`): Repair of toy libraries like the organisers' example (codegen_fix, toy pool). Tasks built from the 42 hand-written toy functions in the style of the organisers' coding_0003 (about 20% of codegen_fix); small, similar to the real competition examples; tests solver prompts tuned to that style. Доля верных в потоке для пула такого вида: 1.0.
- **`nrw_coding_mbpp`** (32 задач, хеш `36635b0a9116`): Implement a function from a description (MBPP). Write a function from a stub and description; unittest checks; short, often one shot; tests the sandbox run, sorter and the tokenizer; no bug to find. Доля верных в потоке для пула такого вида: 0.87.

## Почему столько задач

24-32 на набор (24 там, где пул ограничен или задачи дороги: geometry, hard_symbolic, multi-bug, large file, wtq_list, wtq_big_table). Ориентир: при доле верных 0.7-0.9 стандартная ошибка доли на 30 задачах 7-8 п.п., на 24 около 9; различие между ветвями оценивается на одних и тех же задачах, поэтому шум повторов меньше, но замер ниже (3 повтора) даёт лишь грубую оценку. По 100 задач на поджанр потребовало бы около 3 млн токенов на один проход всех наборов (не по карману). DABench даёт один набор: вне обучения gbm свободны лишь около 50 из 257 задач; `humanevalfix` не вошёл (из 164 задач 159 уже в обучении предиктора).

## Проверки (утечки, пересечения)

Сделано на хосте скриптом `tools/narrow_sets_verify.py` (строит все наборы заново и сверяет хеши):

- Обучающие задачи gbm (`profiles/cost_model.json`, 4726 задач = результаты потока `results.jsonl` ∩ `verify/pred_data.npz` без `verify/lab_task_ids.txt`) исключены через `exclude_ids_file` (`narrow/exclude_ids.txt`, 6032 id: обучение с запасом (4798), `lab_task_ids`, `exclude_all`, все задачи 13 существующих наборов, включая холдаут и `math104_*`). Пересечений с обучением 0.
- С существующими наборами (baseline0, dev_*, mixed100_dev, mixed100_holdout, verify_*40, math16, math104_*): 0. Холдаут не запускался; его id использованы только как исключения.
- Между 24 наборами: 0 общих id (поджанры разведены фильтрами: type, hops, n_bugs, check, последняя цифра id и т. д.); одинаковых текстов вопросов нет (у `codegen_fix` инструкция одинакова во всех задачах, это шаблон организаторов).
- Резерв: 10-16 задач на набор (dabench 10), не пересекается с наборами и между собой; для fresh: спецификация набора с `seed: <reserve_seed>` и `exclude_ids_file: sets/narrow/exclude_with_narrow.txt` даёт задачи резерва (или берите `reserve` из `manifest.json`).
- Информационно: часть задач (30 из 30 у `nrw_data_analysis_dabench`, 9 из 30 у `cg_toy_lib`) когда-то прогонялась в потоке, но не вошла в обучение предиктора (`lab_task_ids` или нет признаков). При переобучении gbm добавьте `narrow/all_task_ids.txt` в файл исключений (`predictor_export.py --exclude`), иначе потоковые прогоны этих задач попадут в обучение.
- Вне этой работы: на хосте `math104_mix` собирается не в свой хеш (frozen `07581b7937f8`, получается `2b38fd311902`; старый `lab.py` даёт то же).

## Шумовой замер (подготовлен, не запущен)

`experiments/noise_nrw_*.yaml` (24 файла): рука `as_is` (как у baseline0: `model: auto`, `HARNESS_FALLBACK: "on"`), 3 повтора, бюджет 100000 (300000 у 5 наборов с оценкой 60k+ токенов), обрезки нет. Порог шума набора = sd очков между повторами (та же логика, что `lab.noise_floor`).

Итого 24 замера: около 1364k токенов (верхняя оценка 2479k), около 7248 запросов шлюза, около 6.2 ч работы диспетчера. Порядок: от дешёвых к дорогим (по столбцу токенов), `nrw_coding_mbpp` последним (оценка самая неопределённая). Три повтора дают грубый sd (±50%); если порог окажется сопоставим с ожидаемым эффектом гипотез, нужен 4-й и 5-й повтор.

## Деплой

На хост копируются только файлы наборов: `experiments/sets/nrw_*.yaml` и `experiments/sets/narrow/*` (в `~/harness-lab/experiments/sets/`). **Пока `lab.py` на хосте не обновлён, наборы там не соберутся**: старый `lab.py` не знает `select` и `exclude_ids_file`, проверка хеша остановит эксперимент (результаты не испортятся). `tools/lab.py` нужно выкатить обычным `deploy.sh` (не `--force`).
