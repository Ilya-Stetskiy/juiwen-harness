# sorter × _all

Блок: Сортировщик очереди (`harness/sorter.py`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: low
- Ожидаемый прирост (очки/прогон): 9.0
- Достоверность: 0.4
- Работа (1-3): 1
- Приоритет: 3.6
- Проверено: 2026-11-11
- Эксперименты: ab_test_cheapfirst, budget_gbm_30k, ab_gbm_vs_placeholder_baseline0_20k
- Потенциал роста [accuracy]: low
- Прирост [accuracy] (очки/прогон): 9.0
- Достоверность [accuracy]: 0.4
- Приоритет [accuracy]: 3.6
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.4
- Приоритет [economy]: нет оценки

## Реализовано
`harness/sorter.py`; `HARNESS_SORTER=greedy|fifo|reward` (по умолчанию `greedy`: награда × p_success / токены p50); поправочный коэффициент по факту; `take_more` для батчера; пропуск по `p_unsolvable` > 0,5. Вариант `CheapFirstScheduler` — ветка автолаборатории `autolab/ab_test_cheapfirst` (в master не влит).

## Измерено
Симулятор (`tools/sim.py`) на 20k/30k/40k: greedy 152/189/207, reward 150/172/192, fifo 131/158/167 очков (sim завышает ≈10% при 30k). Реальный: `ab_test_cheapfirst` (`mixed100_dev`, 30k, 1 повтор): 206 → 215, +9 [−10; +26], clarify. Эффект порядка внутри `budget_gbm_30k` отделён нет: gbm успевает больше задач «за счёт дешёвого порядка».

## Чем ограничено измерение
Один повтор для cheapfirst; sim не калиброван (расхождение с реальностью по `measured`); реального прогона greedy против fifo нет.

## Не проверено
Реальный greedy против fifo; влияние p_success (сейчас заглушка); порядок «по убыванию награды» не доказан (`coding_late_tier`: 132,0 против 133,5); перепроверки в сортировщике.

## Потенциал
Оценка +9 очков (cheapfirst, интервал [−10; +26]), достоверность 0,15: один повтор. Порядок и предиктор нельзя разделить: выигрыш gbm — это во многом выигрыш порядка.

## Связи с другими блоками
Зависит от `predictor` (качество оценок); отдаёт группы в `batcher`; возвращает перепроверки (`recheck`).

## Источники
AUTOLAB_REPORT.md §9.2; harness-gap.md §F S5, §K4; SCRIPTS.md §7

## Следующий шаг
`ab_test_cheapfirst` с 3 повторами на 30k (около 600 запросов).

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | budget_sweep:b30k | бюджет 30k против 20k (BUDGET) | baseline0 · 20k/30k/40k | 2 | очки b20k 179,0 (30 048 ток.), b30k 190,0 (34 334); +11,0 [-5,5; +27,0], в шуме; харнес превысил бюджет (потрачено 150% при 20k, 115% при 30k) | clarify | человек/подсессия (автор в манифесте: human) | нет | autolab/budget_sweep.verdict.md; results/budget_sweep/report.md | оба |
| 2026-11-09 | budget_sweep:b40k | бюджет 40k против 20k (BUDGET) | baseline0 · 20k/30k/40k | 2 | очки b40k 227,0 (58 637 ток., превышение бюджета до 147%), точность 0,755; +48,0 [+25,5; +70,0] против b20k; очки на 1000 токенов: 5,96 / 5,53 / 3,87 | accept | человек/подсессия (автор в манифесте: human) | нет | autolab/budget_sweep.verdict.md; results/budget_sweep/report.md | оба |
| 2026-11-09 | budget_gbm_30k | предиктор placeholder против gbm (HARNESS_PREDICTOR) | baseline0 · 30k | 2 | очки placeholder 201,5 (37 331 ток.) против gbm 226,0 (39 886); +24,5 [-1,0; +51,0], нижняя граница у нуля; завершено задач в пределах 30k: gbm 87 и 90, placeholder 70 и 76 (порядок задач дешевле) | clarify | человек/подсессия (автор в манифесте: human) | нет (HARNESS_PREDICTOR=gbm по умолчанию выключен, commit 76f58f1) | results/budget_gbm_30k/report.md | accuracy |
| 2026-11-09 | sim_sorter_variants | симулятор: greedy, reward, fifo | baseline0 в симуляторе · 20k/30k/40k, оффлайн | нет данных | очки sim greedy 152/189/207, reward 150/172/192, fifo 131/158/167; повтор baseline0 в sim 218 против реальных 219; sim завышает около 10% при 30k; реальный прогон greedy против fifo не делался | clarify | нет данных | нет | AUTOLAB_REPORT.md §9.1; harness-gap.md §F S5 | accuracy |
| 2026-11-10 | budget_sweep | b20k: BUDGET=20000, HARNESS_FALLBACK=on; b30k: BUDGET=30000, HARNESS_FALLBACK=on; b40k: BUDGET=40000, HARNESS_FALLBACK=on | baseline0 · 40000 | 2 | b20k очки 179.0, токены 30048 (эталон); b30k очки 190.0, токены 34334, разность +11.0 [-5.5; 27.0]; b40k очки 227.0, токены 58637, разность +48.0 [25.5; 70.0]; порог 7.69 | accept | автолаборатория | нет | experiments/autolab/budget_sweep.report.md | не указан |
| 2026-11-11 | ab_gbm_vs_placeholder_baseline0_20k | placeholder: HARNESS_PREDICTOR=placeholder; gbm: HARNESS_PREDICTOR=gbm | baseline0 · 20000 | 2 | placeholder очки 147.5, токены 23797 (эталон); gbm очки 186.0, токены 25156, разность +48.0 [20.5; 77.0]; порог 10.0 · режим: accuracy · решение: принята | accept | автолаборатория | нет | нет данных | accuracy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус works-unexplored | автор: status-session
- 2026-11-11 | status: works-unexplored → explored-headroom; confidence: 0.15 → 0.4; experiments: +ab_gbm_vs_placeholder_baseline0_20k | evidence: ab_gbm_vs_placeholder_baseline0_20k: арм gbm +48.0 [20.5, 77.0] очков/прогон против порога 10.0, 2 повт., набор baseline0, вердикт accept | автор: autolab
