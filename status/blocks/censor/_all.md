# censor × _all

Блок: Цензор исполнителя: лимиты и зацикливание (`harness/limits.py`, `harness/detectors.py`)

- Статус: works-unexplored
- Применимость: да
- Потенциал роста: unknown
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.2
- Работа (1-3): 1
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: ab_censor_type_caps_30k, recheck_smoke, recheck_budget, loop_models, ab_censor_type_caps_30k_r4
- Потенциал роста [accuracy]: unknown
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.2
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: low
- Прирост [economy] (% токенов при тех же очках): 1.0
- Достоверность [economy]: 0.3
- Приоритет [economy]: 0.3

## Реализовано
`harness/limits.py`: `HARNESS_CENSOR=off|log|enforce` (по умолчанию `off`), потолок токенов по типу `HARNESS_CENSOR_TOKENS`, `BASIS=fixed|pred`, `FACTOR=3` (рассуждающая ×3), секунды; `enforce` останавливает между вызовами (`CensorStop`); `harness/detectors.py` — зацикливание `HARNESS_LOOP=off|log|retry`.

## Измерено
`ab_censor_type_caps_30k` (`mixed100_dev`, 30k, 2 повтора, `enforce` math=600,search=500,DA=800,coding=700): 217,0 → 216,5, −0,5 [−18,5; +18,0], токены −13% (40 005 → 34 703), clarify. `recheck_budget` (log): помечена 1 из 100 задач. `recheck_smoke` (enforce, потолок math 700): остановлено 4, ни одна не была решена. `loop_models` (16 math): зацикленных 0, оборванных у noreason 3. `ab_censor_type_caps_30k_r4` (4 повтора): 221,8 → 216,2, −5,5 [−18,8; +6,2], clarify, токены на задачу 387,6 против 391,5 (экономия токенов в r4 не подтвердилась).

## Чем ограничено измерение
2 повтора; интервал ±18 (полуширина > порога 7,69); остановка посреди вызова невозможна (ответ приходит целиком), не проверено, отдаёт ли прокси стрим; потолки — прикидки.

## Не проверено
«обрезано зря» на настоящих данных; потолки, подобранные по данным; стрим.

## Потенциал
Нет оценки очков (интервал симметричен около 0). Экономия токенов 13% в `ab_censor_type_caps_30k` не воспроизвелась в r4 (токены на задачу те же); очки не подтверждены.

## Связи с другими блоками
`predictor` (потолок по предсказанию), `budget`, `batcher` (резать пачку по задачам), `fallback`.

## Источники
harness-gap.md §G (S13), §H (B6); AUTOLAB_REPORT.md; verdict на хосте `autolab/ab_censor_type_caps_30k.verdict.md`

## Следующий шаг
Подобрать потолки по распределению стоимости `baseline0`.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | loop_models | noreason против рассуждающей qwen3.6 при HARNESS_LOOP=log | 16 math · без бюджета | 1 | зацикленных 0 у обеих; оборванных по max_tokens: noreason 3 из 18 вызовов (в 1 из 3 задача всё равно решена), reasoning 0; токены на вызов 365 против 997; очки 33,0 против 41,0, +8,0 [0; +18,0], в шуме | clarify | человек/подсессия (автор в манифесте: human) | нет | results/loop_models/report.md | economy |
| 2026-11-09 | recheck_budget | судья + перепроверка сильной моделью (recheck) на полном baseline0 | baseline0 · 100k | 1 | цензор в режиме log при стандартных потолках пометил 1 задачу из 100 (2 710 токенов сверх потолка, не решена) | clarify | человек/подсессия (автор в манифесте: human) | нет | results/recheck_budget/report.md | accuracy |
| 2026-11-10 | ab_censor_type_caps_30k | потолки токенов по типу (math=600, search=500, data_analysis=800, coding=700; CENSOR=enforce) против без потолков | mixed100_dev · 30k | 2 | очки 217,0 (40 005 ток.) против 216,5 (34 703, -13%); -0,5 [-18,5; +18,0], в шуме; полуширина 18,2 > порога | clarify | автолаборатория (orchestrator) | нет | autolab/ab_censor_type_caps_30k.verdict.md; results/ab_censor_type_caps_30k/report.md | economy |
| 2026-11-10 | ab_censor_type_caps_30k_r4 | то же, 4 повтора | mixed100_dev · 30k | 4 | очки 221,8 (точность 0,752; 387,6 ток. на задачу) против 216,2 (0,748; 391,5); -5,5 [-18,8; +6,2], в шуме; вердикт автолаборатории в meta ещё не записан (stage queued), clarify по правилу лаборатории | clarify | автолаборатория (orchestrator) | нет | results/ab_censor_type_caps_30k_r4/report.md | economy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус works-unexplored | автор: status-session
- 2026-11-10 | без изменений статуса | evidence: обновлены измерения по журналу гипотез: ab_fallback_late_30k clarify, ab_censor_type_caps_30k_r4 clarify, ab_guess_* clarify | автор: status-session
- 2026-11-10 | experiments: +ab_censor_type_caps_30k_r4 | evidence: журнал: добавлены эксперименты | автор: status-session
- 2026-11-10 | headroom[economy]: unknown → low; gain[economy]: нет данных → 1.0; confidence[economy]: 0.2 → 0.3 | evidence: ab_censor_type_caps_30k: токены -13% (40 005 -> 34 703) при очках -0,5 в шуме; повтор ab_censor_type_caps_30k_r4 (4 повтора): токены на задачу 387,6 против 391,5 (-1%) при очках -5,5 [-18,8; +6,2]; экономия не воспроизвелась, берётся -1% | режим: economy | автор: status-modes
