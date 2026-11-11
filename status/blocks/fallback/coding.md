# fallback × coding

Блок: Запасной агент (fallback-режимы on / late / off) (`harness/executor.py`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: medium
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.3
- Работа (1-3): 1
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: coding_late_tier, coding_pipeline_vs_hybrid, coding_late_fallback_v2, ab_fallback_off_30k
- Потенциал роста [accuracy]: medium
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.3
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: unknown
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.3
- Приоритет [economy]: нет оценки

## Реализовано
Агент с инструментами решает coding после провала конвейера; невыполнимые закрыты детектором.

## Измерено
`coding_late_tier`: +17,5 очка [+2,0; +36,5]; `baseline0` coding 68,7 против 68,7 (конвейер уже 0,987). `ab_fallback_late_30k` (`mixed100_dev`, 30k, 2 повтора): on 208,0 → late 210,5, +2,5 [−17,0; +21,0], clarify, токены −24% (40 025 → 30 487); `ab_fallback_off_30k` прерван (шлюз остановлен), результатов нет.

## Чем ограничено измерение
Эффект виден только на наборе с 81% точности конвейера; перерасход до 13%.

## Не проверено
Повтор `ab_fallback_late_30k` с 3-4 повторами; `ab_fallback_off_30k` (запасной агент выключен) не завершён.

## Потенциал
Зависит от сложности coding организаторов: нет данных; при лёгком coding 0.

## Связи с другими блоками
`censor` (лимит агента), `solvers/coding`.

## Источники
См. `_all.md` блока.

## Следующий шаг
См. `_all`.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-08 | coding_pipeline_vs_hybrid | конвейер без агента против конвейера + запасной агент (HARNESS_FALLBACK=off/on) | 36 coding codegen_fix · без бюджета | 2 | точность pipeline 0,889 (934 ток. на задачу) против hybrid 0,917 (3053); +2,8 п.п. [-6,9; +12,5], в шуме; на выполнимых +9,4 п.п. [+1,6; +18,8] (SCRIPTS §4) при цене ×2,2; на невыполнимых агент выдумывал исправление | clarify | нет данных (манифест без автора; сессия стенда) | нет (HARNESS_FALLBACK=on по умолчанию) | results/coding_pipeline_vs_hybrid/report.md | accuracy |
| 2026-11-08 | coding_late_fallback | inline без детектора; pipeline+детектор; late+детектор | 36 coding · без бюджета | 2 | точность inline_nodet 0,958 (3052 ток.), pipeline_det 0,889 (810), late_det 0,889 (1609); pipeline_det против inline -6,9 п.п. [-15,3; 0]; на невыполнимых детектор 0 токенов и 100% против 8344 токенов и 87,5%; первая версия late отбрасывала задачи (unfunded), исправлена | clarify | нет данных (манифест без автора; сессия стенда) | нет (late не включён по умолчанию) | results/coding_late_fallback/report.md | accuracy |
| 2026-11-08 | coding_late_fallback_v2 | pipeline+детектор против late+детектор после исправления | 36 coding · без бюджета | 2 | точность pipeline_det 0,917 (796 ток.; 103,0 очка) против late_det 0,861 (1138; 97,5); -5,6 п.п. [-13,9; +2,8], в шуме; причина в разбросе конвейера (провалов 3, 3, 5, 7 по запускам), не в агенте | clarify | нет данных (манифест без автора; сессия стенда) | нет | results/coding_late_fallback_v2/report.md | accuracy |
| 2026-11-08 | coding_late_tier | pipeline против late с порядком по награде (reward) и по порядку (fifo) | 48 coding codegen_fix, температура 0, награды 1-5 · 50k | 2 | очки pipeline 114,5 (точность 0,812; 563 ток.), late_reward 132,0 (0,896; 1040), late_fifo 133,5 (0,896; 1241); late_reward +17,5 [+2,0; +36,5], late_fifo +19,0 [+3,5; +37,5]; ≈1300 токенов за очко; порядок по награде не доказан (132,0 против 133,5); перерасход до 13% (46,6-56,7k при 50k) | accept | нет данных (манифест без автора; сессия стенда) | нет (HARNESS_FALLBACK=late по умолчанию выключен) | results/coding_late_tier/report.md | accuracy |
| 2026-11-09 | baseline0 | as_is (HARNESS_FALLBACK=on) против late; базовая линия смешанного набора | baseline0: 100 задач (25 на тип), награды 1-5, 18% невыполнимых · 100k | 3 | late против as_is по coding: 68,7 против 68,7 очка (0,987 в обоих), токены 657 против 779; изменений нет | clarify | человек/подсессия (автор в манифесте: human) | нет | results/baseline0/report.md | оба |
| 2026-11-10 | ab_fallback_late_30k | fallback on (агент сразу) против late (после очереди) | mixed100_dev · 30k | 2 | очки 208,0 (40 025 ток.) против 210,5 (30 487, -24%); +2,5 [-17,0; +21,0], в шуме; точность 0,70 -> 0,72 | clarify | автолаборатория (orchestrator) | нет | autolab/ab_fallback_late_30k.verdict.md; results/ab_fallback_late_30k/report.md | оба |
| 2026-11-10 | ab_fallback_off_30k | late против off (без запасного агента) | mixed100_dev · 30k | 2 | прерван: шлюз остановлен (429/limits guard), ABORTED.txt, результатов нет; в meta stage queued | failed | автолаборатория (orchestrator) | нет | results/ab_fallback_off_30k/ABORTED.txt | оба |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
- 2026-11-10 | без изменений статуса | evidence: обновлены измерения по журналу гипотез: ab_fallback_late_30k clarify, ab_censor_type_caps_30k_r4 clarify, ab_guess_* clarify | автор: status-session
- 2026-11-10 | experiments: +ab_fallback_off_30k | evidence: журнал: добавлены эксперименты | автор: status-session
