# guess × coding

Блок: Угадывание и сдача нерешаемых (`harness/guess.py`)

- Статус: works-unexplored
- Применимость: да
- Потенциал роста: unknown
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.1
- Работа (1-3): 1
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: ab_guess_free_contradiction_20k, ab_guess_llm_vs_free_20k_coding, ab_skip_contradiction_30k
- Потенциал роста [accuracy]: unknown
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.1
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.1
- Приоритет [economy]: нет оценки

## Реализовано
`free`: ответ `DONE` (файл-маркер, тесты судят репозиторий); для невыполнимых coding этого достаточно, 0 токенов.

## Измерено
`ab_guess_free_contradiction_20k` (`mixed100_dev`, 20k, 2 повтора): пропуск 202,0 против пропуск + `GUESS=free` 196,0, −6,0 [−18,5; +6,5], clarify; `ab_guess_llm_vs_free_20k_coding` (coding, 20k, 1 повтор): free 98,0 против llm 96,0, −2,0 [−6,0; 0], clarify; `ab_skip_contradiction_30k` отсеян симулятором. В лаборатории верный ответ на невыполнимую задачу = отсутствие ответа, поэтому польза угадывания не измерима.

## Чем ограничено измерение
Невыполнимая coding-задача без исправления оценивается как «верно» только в нашей лаборатории.

## Не проверено
Как оценивают организаторы.

## Потенциал
Нет данных.

## Связи с другими блоками
`solvers/coding` (противоречие).

## Источники
См. `_all.md` блока.

## Следующий шаг
См. `_all`.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-10 | ab_skip_contradiction_30k | без детектора против PUNSOLV=contradiction (пропуск coding с противоречием) | mixed100_dev · 30k | нет данных | не запускался: отсеян симулятором (предсказанный эффект меньше 4 очков при каждом бюджете, не виден на фоне шума 7,69) | не запускался | автолаборатория (orchestrator) | нет | autolab/ab_skip_contradiction_30k.meta.json | accuracy |
| 2026-11-10 | ab_guess_free_contradiction_20k | PUNSOLV=contradiction: пропуск против пропуск + GUESS=free (DONE для пропущенных coding) | mixed100_dev · 20k | 2 | очки 202,0 (34 176 ток.) против 196,0 (35 121); -6,0 [-18,5; +6,5], в шуме; точность 0,645 -> 0,640 | clarify | автолаборатория (orchestrator) | нет | autolab/ab_guess_free_contradiction_20k.verdict.md; results/ab_guess_free_contradiction_20k/report.md | accuracy |
| 2026-11-10 | ab_guess_llm_vs_free_20k_coding | GUESS=free против GUESS=llm при PUNSOLV=contradiction | dev_coding (coding) · 20k | 1 | очки 98,0 (57 298 ток.) против 96,0 (48 370); -2,0 [-6,0; 0], в шуме; точность 0,775 -> 0,750; в лаборатории верный ответ на невыполнимую = отсутствие ответа, поэтому польза угадывания тут не измерима | clarify | автолаборатория (orchestrator) | нет | autolab/ab_guess_llm_vs_free_20k_coding.verdict.md; results/ab_guess_llm_vs_free_20k_coding/report.md | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус works-unexplored | автор: status-session
- 2026-11-10 | без изменений статуса | evidence: обновлены измерения по журналу гипотез: ab_fallback_late_30k clarify, ab_censor_type_caps_30k_r4 clarify, ab_guess_* clarify | автор: status-session
- 2026-11-10 | experiments: +ab_guess_free_contradiction_20k,ab_guess_llm_vs_free_20k_coding,ab_skip_contradiction_30k | evidence: журнал: добавлены эксперименты | автор: status-session
