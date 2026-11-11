# recheck × _all

Блок: Перепроверка и возврат в сортировщик (`harness/executor.py`, `HARNESS_RECHECK_*`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: medium
- Ожидаемый прирост (очки/прогон): 1.0
- Достоверность: 0.2
- Работа (1-3): 2
- Приоритет: 0.1
- Проверено: 2026-11-10
- Эксперименты: recheck_budget, ab_recheck_other_math, recheck_later_smoke, recheck_smoke, verify_other
- Потенциал роста [accuracy]: medium
- Прирост [accuracy] (очки/прогон): 1.0
- Достоверность [accuracy]: 0.2
- Приоритет [accuracy]: 0.1
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.2
- Приоритет [economy]: нет оценки

## Реализовано
`HARNESS_VERIFY_ACTION=log|retry|recheck`, `HARNESS_RECHECK_MODE=strong|other`, `HARNESS_RECHECK_QUEUE=inline|later` (возврат в сортировщик, рейтинг reward×(1−p)/цена), `HARNESS_DISAGREE=higher_p|strong|third`, `HARNESS_RECHECK_MAX=1`, `HARNESS_VERIFY_GIVEUP`, `HARNESS_BUDGET_RECHECK=0.10`.

## Измерено
`recheck_budget` (`baseline0`, 100k, 1 повтор, strong inline): 27 перепроверок, 3 замены, очки 219 (те же), проверка 56% расхода. `ab_recheck_other_math` (40 задач, 1 повтор): other+third +1,0 [0; +3], clarify; токены +23% против strong +57%. `recheck_later_smoke` (12 задач): 12 перепроверок, 1 ответ изменён, 29 634 токенов.

## Чем ограничено измерение
Один повтор, 12–100 задач; интервал [0; +3] на 40 задачах не различает эффект.

## Не проверено
`later` против `inline` на объёме; сравнение правил `third`/`higher_p`/`strong`; вердикт `unsolvable` (`GIVEUP` выключен); сколько токенов оставлять под перепроверки.

## Потенциал
+1,0 очка (интервал [0; +3]), достоверность 0,2. Свободных токенов хватает (остаток 44k из 100k при 27 перепроверках), но дорогая перепроверка тем же конвейером почти ничего не даёт.

## Связи с другими блоками
Зависит от `verifier` (кого перепроверять) и `sorter` (когда); тратит `budget`; `guess` для оставшихся.

## Источники
harness-gap.md §H (B2, B3, замер перепроверки), §I; AUTOLAB_REPORT.md §4.2, §4.4

## Следующий шаг
3 повтора `ab_recheck_other_math` на `mixed100_dev` при 30k.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | recheck_smoke | все блоки включены: судья+recheck, цензор enforce (math=700), BUDGET_HARD=on, решатель arith | 16 math · 40k | 1 | точность 0,500, 21,0 очка, потрачено 13,7k из 40k; цензор остановил 4 задачи (ни одна не решена), помечено 7; резерв бюджета работает, лишних отказов нет; дымовой прогон, сравнения нет | clarify | человек/подсессия (автор в манифесте: human) | нет | results/recheck_smoke/report.md | accuracy |
| 2026-11-09 | verification_plan_offline | сигналы free / judge / other / other+judge; политика «судья, при нет другая модель» | порции A+B: math 149, data_analysis 98, search 130 задач, оффлайн | нет данных | AUC (кросс-проверка): math 0,85 / 0,91 / 0,93 / 0,95, data_analysis 0,73 / 0,79 / 0,86 / 0,86, search 0,56 / 0,62 / 0,79 / 0,79; симуляция политики на 100 задач: math +15,4 [+3,4; +30,2] (1620 токенов на очко), data_analysis +31,6 [+14,3; +54,1], search -4,6 [-20,0; +9,2]; симуляция на записанных исходах, лабораторией не подтверждена | clarify | нет данных | да (harness/verifier.py, HARNESS_VERIFY=off по умолчанию) | verification-plan.md §4-§5 | accuracy |
| 2026-11-09 | recheck_budget | судья + перепроверка сильной моделью (recheck) на полном baseline0 | baseline0 · 100k | 1 | очки 219,0 (в baseline0 219 и 232 без верификатора), точность 0,680; проверено 99 ответов, вердикты принять/перепроверка/нерешаема 55/44/0, перепроверок 27, замен 3; потрачено 55 341 из 100 000, проверка 31 065 (56%); неверных осталось 28 против ≈24; AUC судьи: data_analysis 0,51, search 0,91, math 0,71 | clarify | человек/подсессия (автор в манифесте: human) | нет | results/recheck_budget/report.md | accuracy |
| 2026-11-09 | ab_recheck_other_math | log_only, recheck_strong, recheck_other (qwen3.8-27b, DISAGREE=third) | verify_frozen40 · 100k | 1 | очки 96,0 (787 ток. на задачу) / 96,0 (1234, +57%) / 97,0 (970, +23%); strong 0,0 [-13; +12], other +1,0 [0; +3]; перепроверок 8 / 13, замен 1 / 2, неверных 7 / 7 / 5 | clarify | автолаборатория (orchestrator) | нет | results/ab_recheck_other_math/report.md | accuracy |
| 2026-11-09 | recheck_later_smoke | RECHECK_QUEUE=later, RECHECK_MODE=other, DISAGREE=third, RPM_LIMIT=40 (дымовой прогон) | verify_frozen40 · 100k | 1 | 12 отложенных задач, 12 перепроверок завершено, 1 ответ изменён, потрачено 29 634; точность 0,85, 94 очка (против 97 у recheck_other, в шуме); 2 rate_wait, 429 нет | clarify | человек/подсессия (автор в манифесте: human) | нет (HARNESS_RECHECK_QUEUE=inline по умолчанию) | results/recheck_later_smoke/report.md | accuracy |
| 2026-11-09 | verify_other | log_only, judge, other, other_judge (проверка согласием с другой моделью) | verify_frozen40 · 100k | 2 | очки 92,0 / 98,0 / 101,0 / 99,5; other +9,0 [+0,5; +20,0] против порога 7,69 (граничный), judge +6,0 [-1,0; +14,5], other_judge +7,5 [0; +16,5]; токены за прогон 25 477 / 46 657 / 42 591 / 46 480; AUC проверки judge 0,866, other 0,945 | accept | человек/подсессия (автор в манифесте: human) | нет (HARNESS_VERIFY=other по умолчанию выключен) | results/verify_other/report.md | accuracy |
| 2026-11-09 | verify_other_fresh | log_only против other (повтор verify_other на свежем наборе) | verify_fresh40 (20 math, 10 search, 10 data_analysis) · 100k | 3 | очки 119,3 (659 ток. на задачу) против 118,7 (1233, ×1,87); -0,7 [-6,7; +4,7], в шуме; точность 0,900 -> 0,867; не подтвердил verify_other: math +2,3 [0; +6,0], search -3,0 [-7,0; 0] (0,90 -> 0,70), data_analysis 0,0 | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_other_fresh/report.md | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
