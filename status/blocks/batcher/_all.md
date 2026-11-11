# batcher × _all

Блок: Батчер (`harness/batching.py`)

- Статус: explored-headroom
- Применимость: да
- Потенциал роста: medium
- Ожидаемый прирост (очки/прогон): нет данных
- Достоверность: 0.6
- Работа (1-3): 1
- Приоритет: нет оценки
- Проверено: 2026-11-10
- Эксперименты: ab_batch8_auto_30k, ab_batch8_mixed_accuracy_r3, ab_batch_math_noreason_30k, ab_batch8_math104_allmix, ab_batch8_math104_single, batch_math_compute2
- Потенциал роста [accuracy]: medium
- Прирост [accuracy] (очки/прогон): нет данных
- Достоверность [accuracy]: 0.6
- Приоритет [accuracy]: нет оценки
- Потенциал роста [economy]: medium
- Прирост [economy] (% токенов при тех же очках): нет данных
- Достоверность [economy]: 0.6
- Приоритет [economy]: нет оценки

## Реализовано
`harness/batching.py`; `HARNESS_BATCH_SIZE=1|2|4|8`, `HARNESS_BATCH_TYPES=math`, `HARNESS_BATCH_PROMPT=plain|compute`, `HARNESS_BATCH_FALLBACK=single|skip`; `plan(...)` — интерфейс «одна задача или группа»; rpm, тяжесть и остаток бюджета в плане не используются (B8).

## Измерено
См. ячейку `math`. Простой промпт `plain` ломает точность (0,78 → 0,28 при размере 8).

## Чем ограничено измерение
Только math измерен; открытые датасеты.

## Не проверено
Другие типы; размер 16; пачка по предсказанной тяжести; учёт rpm.

## Потенциал
См. `math`: единственный блок со статусом accept в лаборатории на очки (30k).

## Связи с другими блоками
`sorter.take_more`, `predictor`, `budget` (запросы в минуту: пачка ×4–10 меньше запросов), `guard` (инъекция портит пачку).

## Источники
harness-gap.md §G (S11), §H (B8); AUTOLAB_REPORT.md §4.1, §9.2; SCRIPTS.md §7

## Следующий шаг
См. `math`.

## Журнал гипотез
| Дата | Эксперимент | Сравнивалось | Набор · бюджет | Повторы | Результат | Вердикт | Кто предложил | В коде | Отчёт | Режим |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-09 | batch_math | простой промпт: размеры пачки 1, 2, 4, 8 (HARNESS_BATCH_SIZE) | 16 math, noreason · без бюджета | 2 | точность 0,781 / 0,469 / 0,312 / 0,281, токены на задачу 562 / 94 / 66 / 16; очки против b1: b2 -11,5 [-26,0; +3,0], b4 -20,0 [-34,0; -6,0], b8 -22,5 [-36,5; -10,0]; модель печатает догадки без вычислений | reject | человек/подсессия (автор в манифесте: human) | нет (HARNESS_BATCH_PROMPT=plain не используется) | results/batch_math/report.md | оба |
| 2026-11-09 | batch_math_compute | промпт compute (раздел на задачу), размеры 2, 4, 8 | 16 math, noreason · без бюджета | 2 | точность 0,781 / 0,781 / 0,812 / 0,781, токены 487 / 668 / 638 / 987; очки против b1 +1,0 [-6,5; +8,5], +2,5 [0; +7,5], +2,0 [-1,5; +7,5]; лимит ответа 300 токенов на задачу обрезал ответы, 12 из 32 прогонов ушли в одиночный путь | clarify | человек/подсессия (автор в манифесте: human) | нет | results/batch_math_compute/report.md | оба |
| 2026-11-09 | batch_math_compute2 | compute с лимитом 600 токенов на задачу, размеры 4 и 8 | 16 math, noreason · без бюджета | 2 | точность b1 0,750 (558 ток.; 1,25 запроса), b4c 0,906 (512; 0,31), b8c 0,906 (429; 0,12); очки +8,5 [0; +20,0] и +6,5 [+0,5; +14,0] (ниже порога 7,69) | clarify | человек/подсессия (автор в манифесте: human) | нет | results/batch_math_compute2/report.md | оба |
| 2026-11-09 | ab_batch8_mixed_accuracy | b1 против b8c (compute) | verify_frozen40 (20 math, 10 search, 10 data_analysis) · 100k | 1 | очки 95,0 (757 ток. на задачу) против 96,0 (510, -33%); +1,0 [-7; +11], в шуме; по math точность 0,95 -> 1,00, токены 846 -> 514; search 0,6 -> 0,4 (2 задачи, батчится только math) | clarify | автолаборатория (orchestrator) | нет | results/ab_batch8_mixed_accuracy/report.md | оба |
| 2026-11-09 | ab_batch8_mixed_accuracy_r3 | то же, 3 повтора | verify_frozen40 · 100k | 3 | очки 92,3 (28 538 ток. за прогон) против 99,7 (28 642); +7,3 [+2,0; +14,0], ниже порога 7,69; точность 0,792 -> 0,858; токены те же | clarify | автолаборатория (orchestrator) | нет | autolab/ab_batch8_mixed_accuracy_r3.verdict.md; results/ab_batch8_mixed_accuracy_r3/report.md | оба |
| 2026-11-09 | ab_batch_math_noreason_30k | noreason: одиночные против батча 8 compute | mixed100_dev · 30k | 1 | очки 200,0 (34 166 ток.) против 213,0 (35 065); +13,0 [-7; +31], в шуме; токены +3% (на reasoning-модели ранее -33...-39%) | clarify | автолаборатория (orchestrator) | нет | autolab/ab_batch_math_noreason_30k.verdict.md; results/ab_batch_math_noreason_30k/report.md | оба |
| 2026-11-10 | ab_batch8_auto_30k | b1 против b8c (compute, math), модель auto | mixed100_dev · 30k | 2 | очки 192,0 (42 139 ток. за прогон) против 219,5 (35 180, -17%); +27,5 [+11,5; +45,0] против порога 7,69; точность 0,59 -> 0,725 | accept | автолаборатория (orchestrator) | нет (включение: HARNESS_BATCH_SIZE=8 HARNESS_BATCH_PROMPT=compute; решение за человеком) | autolab/ab_batch8_auto_30k.verdict.md; results/ab_batch8_auto_30k/report.md | оба |
| 2026-11-10 | ab_batch8_20k_binding | b1 против b8c при бюджете 20k | mixed100_dev · 20k | 2 | не завершён: в очереди лаборатории (pending) | не завершён | автолаборатория (orchestrator) | нет | autolab/ab_batch8_20k_binding.yaml | оба |
| 2026-11-10 | ab_batch4_vs_batch8_20k | b8c против b4c при бюджете 20k | mixed100_dev · 20k | 2 | не завершён: идёт на хосте (queue/running), отчёта нет | не завершён | автолаборатория (orchestrator) | нет | autolab/ab_batch4_vs_batch8_20k.yaml | оба |

## История
- 2026-11-10 | создано | первичная инвентаризация, статус explored-headroom | автор: status-session
- 2026-11-10 | headroom[economy]: unknown → medium | evidence: токены -17...-33% на math (reasoning), на noreason +3%, на 100k 0%; прирост по типам считается в ячейках типов; в _all число не ставится | режим: economy | автор: status-modes
