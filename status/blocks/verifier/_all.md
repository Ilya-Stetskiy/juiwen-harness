# verifier × _all

Блок: Верификатор: уровни free / judge / other (`harness/judge.py`, `harness/verifier.py`, `harness/search_check.py`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: medium
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.3
- Работа (1-3): 2
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: verify_other, verify_other_fresh, verify_frozen_v2, verify_fresh_v2, recheck_budget, ab_verify_types_selective
- Потенциал роста [accuracy]: medium
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.3
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: low
- Прирост [economy] (% токенов при тех же очках): 11.0
- Достоверность [economy]: 0.2
- Приоритет [economy]: 1.1

## Реализовано
`harness/judge.py` + `verifier.py` + `profiles/verifier_model.json` (answer-verifier); `HARNESS_VERIFY=off|free|judge|other|other_judge` (по умолчанию `off`), `HARNESS_VERIFY_JUDGE_TYPES=math,data_analysis,coding`, `_OTHER_TYPES`, `HARNESS_VERIFY_MIN`/`_ACCEPT`; вердикт accept/recheck/unsolvable. `search_check.py` (проверка по источнику) — оффлайн, как уровень не подключён.

## Измерено
Оффлайн (verification-plan): AUC free/judge/other: math 0,85/0,91/0,93, DA 0,73/0,79/0,86, search 0,56/0,62/0,79. Лаборатория: `verify_other` (frozen40, 2 повтора): other +9,0 [+0,5; +20,0] **accept**, токены +67%; `verify_other_fresh` (fresh40, 3 повтора): other −0,7 [−6,7; +4,7], токены ×1,87 — **не подтвердился**. `verify_frozen_v2` AUC 0,93 против `verify_fresh_v2` 0,58 (3–6 неверных): подозрение на переобучение под известные наборы.

## Чем ограничено измерение
Неверных ответов 3–12 на эксперимент; интервалы AUC ±0,1; открытые данные; coding не измерялся; проверка стоит заметную долю бюджета (56% расхода в `recheck_budget`).

## Не проверено
Эффект при бюджете 30k; порог принятия по очкам; свежий набор из 30+ неверных; задачи организаторов.

## Потенциал
Прирост очков не установлен: один accept (frozen) против «в шуме» (fresh). Организаторы считают проверки второстепенными при нехватке бюджета; цена +67–87% токенов.

## Связи с другими блоками
Выход в `recheck`, `sorter` (p), `guess`/`unsolvable`; потребляет `budget`; `solvers` (доверие `final` минует проверку).

## Источники
verification-plan.md §1–§7; harness-gap.md §G (S10), §H, §K1, §J-verify_other; AUTOLAB_REPORT.md §8

## Следующий шаг
Набрать размеченные неверные ответы дёшево на локальной модели (`verify_collect_multi`, уже 2 128 строк) и оценить пороги; не гонять шлюзовые A/B.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | verify_frozen | заглушка проверки (judge, placeholder+judge) на замороженном наборе | verify_frozen40 · без бюджета | 1 | AUC 0,49 (33 ответа, 5 неверных; поймано 2 из 5, ложных тревог 43%), 18 токенов на задачу; метрика AUC, не очки; по search AUC 0,25 | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_frozen/report.md | accuracy |
| 2026-11-09 | verify_fresh | та же заглушка на свежем наборе | verify_fresh40 · без бюджета | 1 | AUC 0,60 (31 ответ, 3 неверных; поймано 2 из 3, ложных тревог 46%), 18 токенов на задачу; разрыв 0,49/0,60 в шуме | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_fresh/report.md | accuracy |
| 2026-11-09 | verify_frozen_v2 | модель answer-verifier (судья + бесплатный уровень) на замороженном наборе | verify_frozen40 · без бюджета | 1 | AUC 0,93 (33 ответа, 6 неверных; бесплатная часть 0,92; поймано 2 из 6, ложных тревог 7%), 66 токенов на задачу; метрика AUC, не очки | clarify | человек/подсессия (автор в манифесте: human) | да (harness/verifier.py, profiles/verifier_model.json, коммит 18c3e2d) | results/verify_frozen_v2/report.md | accuracy |
| 2026-11-09 | verify_fresh_v2 | модель answer-verifier на свежем наборе | verify_fresh40 · без бюджета | 1 | AUC 0,58 (31 ответ, 3 неверных; бесплатная часть 0,57; поймано 0 из 3, ложных тревог 0), 77 токенов на задачу (search 150-170); разрыв 0,93 против 0,58 по 3-6 неверным не доказан, подозрение на переобучение под известные наборы | clarify | человек/подсессия (автор в манифесте: human) | да (harness/verifier.py, profiles/verifier_model.json, коммит 18c3e2d) | results/verify_fresh_v2/report.md | accuracy |
| 2026-11-09 | verification_plan_offline | сигналы free / judge / other / other+judge; политика «судья, при нет другая модель» | порции A+B: math 149, data_analysis 98, search 130 задач, оффлайн | нет данных | AUC (кросс-проверка): math 0,85 / 0,91 / 0,93 / 0,95, data_analysis 0,73 / 0,79 / 0,86 / 0,86, search 0,56 / 0,62 / 0,79 / 0,79; симуляция политики на 100 задач: math +15,4 [+3,4; +30,2] (1620 токенов на очко), data_analysis +31,6 [+14,3; +54,1], search -4,6 [-20,0; +9,2]; симуляция на записанных исходах, лабораторией не подтверждена | clarify | нет данных | да (harness/verifier.py, HARNESS_VERIFY=off по умолчанию) | verification-plan.md §4-§5 | accuracy |
| 2026-11-09 | ab_verify_types_selective | судья на всех типах против судьи только на math и data_analysis (HARNESS_VERIFY_JUDGE_TYPES) | verify_frozen40 · 100k | 1 | очки 96,0 (1050 ток. на задачу) против 100,0 (933); +4,0 [0; +11,0], в шуме; по сути A/A (типы по умолчанию уже без search), токены проверки около 100 на ответ против 314; точность 0,825 -> 0,875 | clarify | автолаборатория (orchestrator) | да (JUDGE_TYPES по умолчанию без search, коммит 1d77a4f) | results/ab_verify_types_selective/report.md | оба |
| 2026-11-09 | recheck_budget | судья + перепроверка сильной моделью (recheck) на полном baseline0 | baseline0 · 100k | 1 | очки 219,0 (в baseline0 219 и 232 без верификатора), точность 0,680; проверено 99 ответов, вердикты принять/перепроверка/нерешаема 55/44/0, перепроверок 27, замен 3; потрачено 55 341 из 100 000, проверка 31 065 (56%); неверных осталось 28 против ≈24; AUC судьи: data_analysis 0,51, search 0,91, math 0,71 | clarify | человек/подсессия (автор в манифесте: human) | нет | results/recheck_budget/report.md | accuracy |
| 2026-11-09 | verify_other | log_only, judge, other, other_judge (проверка согласием с другой моделью) | verify_frozen40 · 100k | 2 | очки 92,0 / 98,0 / 101,0 / 99,5; other +9,0 [+0,5; +20,0] против порога 7,69 (граничный), judge +6,0 [-1,0; +14,5], other_judge +7,5 [0; +16,5]; токены за прогон 25 477 / 46 657 / 42 591 / 46 480; AUC проверки judge 0,866, other 0,945 | accept | человек/подсессия (автор в манифесте: human) | нет (HARNESS_VERIFY=other по умолчанию выключен) | results/verify_other/report.md | accuracy |
| 2026-11-09 | verify_other_fresh | log_only против other (повтор verify_other на свежем наборе) | verify_fresh40 (20 math, 10 search, 10 data_analysis) · 100k | 3 | очки 119,3 (659 ток. на задачу) против 118,7 (1233, ×1,87); -0,7 [-6,7; +4,7], в шуме; точность 0,900 -> 0,867; не подтвердил verify_other: math +2,3 [0; +6,0], search -3,0 [-7,0; 0] (0,90 -> 0,70), data_analysis 0,0 | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_other_fresh/report.md | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
- 2026-11-10 | headroom[economy]: unknown → low; gain[economy]: нет данных → 11.0; confidence[economy]: 0.3 → 0.2 | evidence: ab_verify_types_selective: судья только на math и data_analysis вместо всех типов, токены на задачу -11% (1050 -> 933) при очках +4,0 [0; +11] (в шуме), 1 повтор, 100k; по сути A/A, выводы слабые | режим: economy | автор: status-modes
