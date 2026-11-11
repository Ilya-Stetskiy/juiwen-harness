# sorter × _all

Блок: Сортировщик очереди (`harness/sorter.py`)

- Статус: works-unexplored
- Применимость: да
- Потенциал роста: low
- Ожидаемый прирост (очки/прогон): 9.0
- Достоверность: 0.15
- Работа (1-3): 1
- Приоритет: 1.35
- Проверено: 2026-11-10
- Эксперименты: ab_test_cheapfirst, budget_gbm_30k
- Потенциал роста [accuracy]: low
- Прирост [accuracy] (очки/прогон): 9.0
- Достоверность [accuracy]: 0.15
- Приоритет [accuracy]: 1.35
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.15
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
| 2026-11-09 | ab_test_cheapfirst | сортировщик greedy против cheapfirst (код от модели) | mixed100_dev · 30k | 1 | очки 206 (36 186 ток.) против 215 (36 522); +9,0 [-10; +26], в шуме; точность 0,66 -> 0,76; sim pass (-12,3/-7,7/+2,9) | clarify | автолаборатория (orchestrator) | ветка:autolab/ab_test_cheapfirst (1c36c4a, в master не влита) | autolab/ab_test_cheapfirst.verdict.md; results/ab_test_cheapfirst/report.md | accuracy |
| 2026-11-09 | sim_sorter_variants | симулятор: greedy, reward, fifo | baseline0 в симуляторе · 20k/30k/40k, оффлайн | нет данных | очки sim greedy 152/189/207, reward 150/172/192, fifo 131/158/167; повтор baseline0 в sim 218 против реальных 219; sim завышает около 10% при 30k; реальный прогон greedy против fifo не делался | clarify | нет данных | нет | AUTOLAB_REPORT.md §9.1; harness-gap.md §F S5 | accuracy |
| 2026-11-10 | mode_sweep_30k | legacy / accuracy / economy | baseline0 · 30k | 3 | economy vs legacy +21,0 [-1,7; +43,7] (clarify), accuracy -10,7 [-37; +17] (clarify); точность 0,76 против 0,64; economy тратит 45k при бюджете 30k (150 %) | clarify | Claude | нет | нет данных | economy |
| 2026-11-10 | mode_sweep_60k | legacy / accuracy / economy / auto | baseline0 · 60k | 2 | economy +14,5 [-1; +31], auto +8,5 [-5; +22], accuracy +4,0 [-9,5; +19] против legacy (все clarify); auto выбрал economy; economy 4,38 очка/1k токенов против 3,35 | clarify | Claude | нет | нет данных | economy |
| 2026-11-10 | mode_confirm_60k | legacy / economy | baseline0 · 60k | 3 | economy vs legacy +13,7 [-8,3; +36,7] (clarify), расход 0,80-0,94 бюджета против 1,04-1,16 | clarify | Claude | нет | нет данных | economy |
| 2026-11-10 | mode_mixed_30k | legacy / economy | mixed100_dev · 30k | 3 | economy vs legacy +39,3 [+16,7; +62,3] (accept), точность 0,77 против 0,57, токены -16 % | accept | Claude | нет | нет данных | economy |
| 2026-11-10 | mode_mixed_60k | legacy / economy | mixed100_dev · 60k | 3 | economy vs legacy -9,7 [-20,7; +0,3] (clarify): economy хуже при 60k на этом наборе | clarify | Claude | нет | нет данных | economy |
| 2026-11-11 | parts_baseline0_60k | legacy / gbm / батчи / gbm+батчи / economy | baseline0 · 60k | 3 | economy +6,7 [-12,0; +25,3], батчи +7,0, gbm -1,0, gbm+батчи +6,7: все в шуме; токены legacy 63,9k -> economy 51,4k | clarify | Claude | нет | нет данных | economy |
| 2026-11-11 | parts_mixed_60k | legacy / gbm / батчи / gbm+батчи / economy | mixed100_dev · 60k | 3 | economy -8,3 [-24,0; +5,7] (в этот раз тоже ниже), gbm+батчи -5,0, батчи +2,0, gbm +4,0 (но токены 72k против 55k): в шуме | clarify | Claude | нет | нет данных | economy |
| 2026-11-11 | full_baseline0_100k | legacy / economy / accuracy | baseline0 · 100k | 3 | accuracy +15,3 [-4,0; +34,7] за +24 % токенов (78,7k против 63,7k), economy +4,0 [-16,3; +24,7] | clarify | Claude | нет | нет данных | accuracy |
| 2026-11-11 | full_mixed_100k | legacy / economy / accuracy | mixed100_dev · 100k | 3 | economy -11,3 [-29,0; +3,0], accuracy -3,3 [-20,7; +12,7] при +57 % токенов (92,7k против 59,0k) | clarify | Claude | нет | нет данных | economy |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус works-unexplored | автор: status-session
