# recheck × data_analysis

Блок: Перепроверка и возврат в сортировщик (`harness/executor.py`, `HARNESS_RECHECK_*`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: medium
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.2
- Работа (1-3): 2
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: recheck_budget, ab_recheck_other_math, verify_other_fresh
- Потенциал роста [accuracy]: medium
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.2
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.2
- Приоритет [economy]: нет оценки

## Реализовано
Судья + перепроверка другой моделью.

## Измерено
`recheck_budget`: DA AUC судьи 0,51, замен почти нет; fresh40: DA ±0.

## Чем ограничено измерение
Мало неверных.

## Не проверено
Что мешает исправлению: вторая попытка ошибается там же.

## Потенциал
Оффлайн-симуляция +31,6 на 100 задач, лаборатория 0; не разобрано.

## Связи с другими блоками
`verifier/data_analysis`.

## Источники
См. `_all.md` блока.

## Следующий шаг
См. `verifier/data_analysis`.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | recheck_later_smoke | RECHECK_QUEUE=later, RECHECK_MODE=other, DISAGREE=third, RPM_LIMIT=40 (дымовой прогон) | verify_frozen40 · 100k | 1 | 12 отложенных задач, 12 перепроверок завершено, 1 ответ изменён, потрачено 29 634; точность 0,85, 94 очка (против 97 у recheck_other, в шуме); 2 rate_wait, 429 нет | clarify | человек/подсессия (автор в манифесте: human) | нет (HARNESS_RECHECK_QUEUE=inline по умолчанию) | results/recheck_later_smoke/report.md | accuracy |
| 2026-11-09 | verify_other | log_only, judge, other, other_judge (проверка согласием с другой моделью) | verify_frozen40 · 100k | 2 | data_analysis: AUC 1,00 / 0,97, изменений точности нет; остальное см. _all | accept | человек/подсессия (автор в манифесте: human) | нет (HARNESS_VERIFY=other по умолчанию выключен) | results/verify_other/report.md | accuracy |
| 2026-11-09 | verify_other_fresh | log_only против other (повтор verify_other на свежем наборе) | verify_fresh40 (20 math, 10 search, 10 data_analysis) · 100k | 3 | data_analysis 0,0 [0; 0]; остальное см. _all | clarify | человек/подсессия (автор в манифесте: human) | нет | results/verify_other_fresh/report.md | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
