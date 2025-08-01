Стандартный DataLoader с shuffle=True просто берет случайные N элементов из датасета. При использовании нашего hierarchical_collate_fn мы надеемся, что в этой случайной выборке окажутся нужные нам комбинации сигналов (из одного замера, из разных замеров одного элемента, из разных элементов).

Проблемы такого подхода:

Нестабильность: В одном батче может случайно оказаться много полезных пар, а в другом — почти ни одной. Это приводит к "шумным" градиентам: то они большие и полезные, то почти нулевые (когда loss_metric равен 0 из-за отсутствия триплетов). Обучение становится нестабильным.
Неэффективность: Каждый раз, когда collate_fn не может сформировать триплеты, мы либо пропускаем батч, либо обучаемся только на одной части функции потерь (loss_kp). Это впустую потраченные вычислительные ресурсы.
Плохая сходимость: Модель не получает стабильного сигнала для обучения эмбеддинг-пространства, что может замедлить или даже остановить сходимость метрической части потерь.
Решение — Семплер "P-K":

Семплер решает эту проблему, изменяя сам принцип формирования батча. Вместо случайного выбора отдельных сигналов, он работает на уровне классов.

P (Persons/Elements): Количество уникальных классов (в нашем случае, измерительных элементов), которые мы хотим видеть в каждом батче.
K (Samples): Количество примеров (сигналов) для каждого из этих P классов.
Таким образом, каждый батч будет иметь гарантированный размер P * K и структуру, которая идеально подходит для нашей задачи:

Гарантированные негативы на уровне элемента: Так как в батче всегда P > 1 элементов, для любого сигнала (якоря) из элемента A мы гарантированно найдем негативные примеры из элементов B, C, и т.д.
Высокая вероятность позитивов и негативов на уровне замера: Выбирая K > 1 сигналов из одного элемента, мы резко повышаем вероятность того, что среди них окажутся сигналы как из одного замера (для anchor-positive пар), так и из разных замеров (для anchor-session_negative пар).
Это делает каждый батч полезным, градиенты — стабильными, а обучение — значительно более эффективным.