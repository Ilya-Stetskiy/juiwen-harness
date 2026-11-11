# verifier × data_analysis

Блок: Верификатор: уровни free / judge / other (`harness/judge.py`, `harness/verifier.py`, `harness/search_check.py`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: medium
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.3
- Работа (1-3): 2
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: verify_other, verify_other_fresh, recheck_budget
- Потенциал роста [accuracy]: medium
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.3
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.3
- Приоритет [economy]: нет оценки

## Реализовано
Судья (≈75–200 токенов) + `other` (≈240–440 токенов); free-признаки.

## Измерено
Оффлайн: free 0,73, judge 0,79 [0,67–0,90], other 0,86 [0,76–0,94]; `verify_other`: AUC 1,00 (2 неверных). `recheck_budget`: AUC DA 0,51 (судья, 100 задач). Лаборатория fresh40: DA +0,0 [0; 0]. Оффлайн-симуляция политики: +31,6 [+14,3; +54,1] очка на 100 задач DA, но реальный замер не подтвердил.

## Чем ограничено измерение
Неверных 2–36; симуляция политики на записанных исходах.

## Не проверено
Стоимость 330 токенов на ответ против пользы при 30k.

## Потенциал
Оффлайн большой (+31,6 на 100 DA-задач), в лаборатории 0: расхождение не разобрано.

## Связи с другими блоками
`recheck/data_analysis`, `executor/data_analysis`.

## Источники
См. `_all.md` блока.

## Следующий шаг
Разобрать расхождение симуляции и fresh40 (почему 0 при AUC 0,86).

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | verification_plan_offline | сигналы free / judge / other / other+judge; политика «судья, при нет другая модель» | порции A+B: math 149, data_analysis 98, search 130 задач, оффлайн | нет данных | AUC free 0,73, judge 0,79 [0,67; 0,90], other 0,86 [0,76; 0,94]; симуляция политики +31,6 [+14,3; +54,1] на 100 data_analysis (оффлайн); лабораторные fresh40 дали 0,0, расхождение не разобрано | accept | нет данных | да (harness/verifier.py, HARNESS_VERIFY=off по умолчанию) | verification-plan.md §4-§5 | accuracy |
| 2026-11-09 | ab_verify_types_selective | судья на всех типах против судьи только на math и data_analysis (HARNESS_VERIFY_JUDGE_TYPES) | verify_frozen40 · 100k | 1 | очки 96,0 (1050 ток. на задачу) против 100,0 (933); +4,0 [0; +11,0], в шуме; по сути A/A (типы по умолчанию уже без search), токены проверки около 100 на ответ против 314; точность 0,825 -> 0,875 | clarify | автолаборатория (orchestrator) | да (JUDGE_TYPES по умолчанию без search, коммит 1d77a4f) | results/ab_verify_types_selective/report.md | оба |
| 2026-11-09 | recheck_budget | судья + перепроверка сильной моделью (recheck) на полном baseline0 | baseline0 · 100k | 1 | AUC судьи в этом прогоне 0,51; остальное см. _all | clarify | человек/подсессия (автор в манифесте: human) | нет | results/recheck_budget/report.md | accuracy |
| 2026-11-09 | verify_other | log_only, judge, other, other_judge (проверка согласием с другой моделью) | verify_frozen40 · 100k | 2 | AUC other 1,00 (2 неверных), other_judge 0,97; остальное см. _all | accept | человек/подсессия (автор в манифесте: human) | нет (HARNESS_VERIFY=other по умолчанию выключен) | results/verify_other/report.md | accuracy |
| 2026-11-09 | verify_other_fresh | log_only против other (повтор verify_other на свежем наборе) | verify_fresh40 (20 math, 10 search, 10 data_analysis) · 100k | 3 | data_analysis 0,0 [0; 0] (одинаково на 10 задачах); остальное см. _all | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_other_fresh/report.md | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
