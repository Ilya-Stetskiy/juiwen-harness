# recheck × math

Блок: Перепроверка и возврат в сортировщик (`harness/executor.py`, `HARNESS_RECHECK_*`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: low
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.2
- Работа (1-3): 2
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: ab_recheck_other_math, recheck_later_smoke
- Потенциал роста [accuracy]: low
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.2
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.2
- Приоритет [economy]: нет оценки

## Реализовано
Судья на math: порог 0,8; перепроверка другой моделью (`qwen3.8-27b`) с третьим мнением.

## Измерено
`ab_recheck_other_math` (math+DA, 40 задач): 13 перепроверок, 2 замены; точность 0,825 → 0,850.

## Чем ограничено измерение
2 задачи из 40.

## Не проверено
Повторы.

## Потенциал
Малый: неверных мало.

## Связи с другими блоками
`verifier/math`.

## Источники
См. `_all.md` блока.

## Следующий шаг
См. `_all`.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | recheck_smoke | все блоки включены: судья+recheck, цензор enforce (math=700), BUDGET_HARD=on, решатель arith | 16 math · 40k | 1 | точность 0,500, 21,0 очка, потрачено 13,7k из 40k; цензор остановил 4 задачи (ни одна не решена), помечено 7; резерв бюджета работает, лишних отказов нет; дымовой прогон, сравнения нет | clarify | человек/подсессия (автор в манифесте: human) | нет | results/recheck_smoke/report.md | accuracy |
| 2026-11-09 | recheck_later_smoke | RECHECK_QUEUE=later, RECHECK_MODE=other, DISAGREE=third, RPM_LIMIT=40 (дымовой прогон) | verify_frozen40 · 100k | 1 | 12 отложенных задач, 12 перепроверок завершено, 1 ответ изменён, потрачено 29 634; точность 0,85, 94 очка (против 97 у recheck_other, в шуме); 2 rate_wait, 429 нет | clarify | человек/подсессия (автор в манифесте: human) | нет (HARNESS_RECHECK_QUEUE=inline по умолчанию) | results/recheck_later_smoke/report.md | accuracy |
| 2026-11-09 | verify_other | log_only, judge, other, other_judge (проверка согласием с другой моделью) | verify_frozen40 · 100k | 2 | math: точность 0,925 -> 0,975 при замене ответов; остальное см. _all | accept | человек/подсессия (автор в манифесте: human) | нет (HARNESS_VERIFY=other по умолчанию выключен) | results/verify_other/report.md | accuracy |
| 2026-11-09 | verify_other_fresh | log_only против other (повтор verify_other на свежем наборе) | verify_fresh40 (20 math, 10 search, 10 data_analysis) · 100k | 3 | math +2,3 [0; +6,0]; остальное см. _all | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_other_fresh/report.md | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
