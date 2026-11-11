# verifier × math

Блок: Верификатор: уровни free / judge / other (`harness/judge.py`, `harness/verifier.py`, `harness/search_check.py`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: low
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.3
- Работа (1-3): 1
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: verify_other, verify_other_fresh, ab_verify_types_selective, ab_recheck_other_math
- Потенциал роста [accuracy]: low
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.3
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.3
- Приоритет [economy]: нет оценки

## Реализовано
Судья «вычислила ли программа ответ» (≈46 токенов), free-признаки (`retry_used`, стоимость), `other` (≈470 токенов).

## Измерено
AUC free 0,85, judge 0,91 [0,82–0,99], other 0,93; ловит 23 из 26 неверных при 25% пометок. Лаборатория: fresh40 math +2,3 [0,0; +6,0] очка (20 задач, 3 повтора), в шуме; в `verify_other` frozen неверных по math нет.

## Чем ограничено измерение
Неверных мало (26 всего); данные смещены в сложные источники MATH.

## Не проверено
Цена/польза при 30k.

## Потенциал
Малый: точность math на свежем наборе 0,90, неверных мало; судья дёшев (≈46 токенов).

## Связи с другими блоками
`recheck/math`, `batcher/math`.

## Источники
См. `_all.md` блока.

## Следующий шаг
Нет, пока не ясно, что ошибок достаточно для выгоды.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | verification_plan_offline | сигналы free / judge / other / other+judge; политика «судья, при нет другая модель» | порции A+B: math 149, data_analysis 98, search 130 задач, оффлайн | нет данных | AUC free 0,85, judge 0,91 [0,82; 0,99], other 0,93; судья находит 23 из 26 неверных при 25% пометок; симуляция политики +15,4 [+3,4; +30,2] на 100 math (оффлайн, не лабораторный accept) | accept | нет данных | да (harness/verifier.py, HARNESS_VERIFY=off по умолчанию) | verification-plan.md §4-§5 | accuracy |
| 2026-11-09 | recheck_budget | судья + перепроверка сильной моделью (recheck) на полном baseline0 | baseline0 · 100k | 1 | AUC судьи 0,71 (2 неверных); остальное см. _all | clarify | человек/подсессия (автор в манифесте: human) | нет | results/recheck_budget/report.md | accuracy |
| 2026-11-09 | verify_other | log_only, judge, other, other_judge (проверка согласием с другой моделью) | verify_frozen40 · 100k | 2 | по math неверных ответов в other нет; точность math 0,925 -> 0,975 (other); остальное см. _all | accept | человек/подсессия (автор в манифесте: human) | нет (HARNESS_VERIFY=other по умолчанию выключен) | results/verify_other/report.md | accuracy |
| 2026-11-09 | verify_other_fresh | log_only против other (повтор verify_other на свежем наборе) | verify_fresh40 (20 math, 10 search, 10 data_analysis) · 100k | 3 | math +2,3 [0; +6,0] очка на 20 задач, в шуме; остальное см. _all | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_other_fresh/report.md | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
