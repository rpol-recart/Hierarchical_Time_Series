Отличный следующий шаг! Добавление слоя внимания (Attention Mechanism) — это одно из самых мощных и популярных улучшений для моделей, работающих с последовательностями. Оно не только способно повысить качество модели, но и решает главную задачу интерпретируемости, о которой вы спросили.

### Что такое Слой Внимания и Что он Привнесет?

Представьте, как вы читаете длинное предложение, чтобы ответить на вопрос. Вы не держите в голове каждое слово с одинаковой "важностью". Ваш мозг интуитивно концентрируется на ключевых словах, которые наиболее релевантны для ответа.

**Слой внимания делает то же самое для нейросети.**

В вашей текущей модели, для получения эмбеддинга мы берем *последнее скрытое состояние GRU*. Это заставляет модель "впихнуть" всю информацию о сигнале в этот один-единственный вектор, что является узким местом, особенно для длинных сигналов.

Слой внимания исправляет это. Вместо того чтобы брать последнее состояние, он:
1.  Просматривает **все** скрытые состояния GRU на каждом временном шаге.
2.  **Вычисляет веса "важности"** для каждого шага.
3.  Создает итоговый вектор (называемый "контекстный вектор") как **взвешенную сумму** всех скрытых состояний.

#### Ключевые Преимущества:

1.  **Повышение Качества (Performance):** Модель больше не ограничена последним состоянием и может динамически выбирать, какая часть сигнала важна для формирования эмбеддинга. Это часто приводит к более точным и робастным представлениям.
2.  **Встроенная Интерпретируемость:** Самое главное для вашего вопроса! **Веса внимания — это и есть готовая карта значимости.** Нам больше не нужны сложные градиентные методы. Мы можем просто взять эти веса и визуализировать их, чтобы увидеть, на какие участки сигнала модель "обратила внимание" при создании эмбеддинга. Это прямой и чистый способ понять решение модели.
3.  **Лучшая Работа с Длинными Сигналами:** Механизм внимания помогает бороться с проблемой "забывания" информации в начале последовательности, которая свойственна рекуррентным сетям.

---

### Как Добавить Слой Внимания в Модель (Код)

Мы реализуем классический аддитивный механизм внимания.

#### Шаг 1: Создание Отдельного Модуля Внимания

Хорошей практикой является вынесение слоя внимания в отдельный класс `nn.Module`. Это делает код чище и позволяет его переиспользовать.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Аддитивный механизм внимания.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # Слой, который вычисляет "энергию" или "оценку" для каждого скрытого состояния
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, gru_output):
        """
        Args:
            gru_output (torch.Tensor): Выход GRU слоя. 
                                       Размер: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context_vector (torch.Tensor): Контекстный вектор, взвешенная сумма выходов GRU.
                                           Размер: (batch_size, hidden_dim)
            attention_weights (torch.Tensor): Веса внимания.
                                              Размер: (batch_size, seq_len)
        """
        # Прогоняем выходы GRU через слой для получения оценок
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, 1)
        attn_energies = self.attn_layer(gru_output)
        
        # Применяем Softmax по временной оси (dim=1) для получения весов
        # .squeeze(-1) убирает последнюю размерность: (batch, seq_len, 1) -> (batch, seq_len)
        attention_weights = F.softmax(attn_energies, dim=1).squeeze(-1)
        
        # Чтобы посчитать взвешенную сумму, нам нужно умножить выходы GRU на веса.
        # Для этого веса нужно расширить обратно до (batch, seq_len, 1)
        expanded_weights = attention_weights.unsqueeze(-1)
        
        # Умножаем каждый выход GRU на его вес внимания
        # (batch, seq_len, hidden_dim) * (batch, seq_len, 1) -> (batch, seq_len, hidden_dim)
        weighted_output = gru_output * expanded_weights
        
        # Суммируем по временной оси для получения контекстного вектора
        # (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        context_vector = torch.sum(weighted_output, dim=1)
        
        return context_vector, attention_weights
```

#### Шаг 2: Интеграция Внимания в Основную Модель

Теперь мы создадим новую версию нашей модели `HierarchicalSignalNet`, которая использует этот слой.

```python
class HierarchicalSignalNet_WithAttention(nn.Module):
    def __init__(self, input_dim=1, embedding_dim=64, rnn_hidden_size=128, num_keypoints=5):
        super(HierarchicalSignalNet_WithAttention, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        
        # Сверточный ствол (остается без изменений)
        self.backbone = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Рекуррентный слой (остается без изменений)
        self.gru = nn.GRU(
            input_size=64, 
            hidden_size=rnn_hidden_size, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # --- ИЗМЕНЕНИЕ 1: Добавляем наш слой внимания ---
        # Входной размер для внимания равен rnn_hidden_size * 2, так как GRU двунаправленный
        self.attention = Attention(hidden_dim=rnn_hidden_size * 2)

        # --- ИЗМЕНЕНИЕ 2: Голова для эмбеддингов теперь принимает вектор той же размерности ---
        self.embedding_head = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, embedding_dim),
            # Сюда можно добавить Dropout или BatchNorm, если нужно
        )
        
        # Голова для ключевых точек (остается без изменений)
        self.keypoint_head = nn.Linear(rnn_hidden_size * 2, num_keypoints)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.backbone(x)
        
        # (batch, channels, seq_len) -> (batch, seq_len, channels) для GRU
        x = x.permute(0, 2, 1)
        
        gru_output, _ = self.gru(x)
        # gru_output: (batch, seq_len, rnn_hidden_size * 2)
        
        # --- ИЗМЕНЕНИЕ 3: Применяем внимание ---
        # Вместо того чтобы брать последнее состояние, мы вычисляем контекстный вектор
        context_vector, attention_weights = self.attention(gru_output)
        
        # --- ИЗМЕНЕНИЕ 4: Формируем выходы ---
        # Эмбеддинг создается из контекстного вектора
        embedding = self.embedding_head(context_vector)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # Ключевые точки по-прежнему предсказываются для каждого шага из полного выхода GRU
        keypoint_logits = self.keypoint_head(gru_output)
        
        # Возвращаем веса внимания для интерпретации!
        return embedding, keypoint_logits, attention_weights
```

### Как это Использовать и Визуализировать?

Теперь ваш цикл обучения и инференса немного изменится, так как модель возвращает три значения. При инференсе вы можете легко визуализировать внимание.

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(original_signal, attention_weights, title="Карта внимания (Attention Map)"):
    """
    Визуализирует сигнал и веса внимания, которые модель на него наложила.
    """
    fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. График исходного сигнала
    ax[0].plot(original_signal, color='royalblue', label='Исходный сигнал')
    ax[0].set_title(title)
    ax[0].set_ylabel("Амплитуда")
    ax[0].grid(True, linestyle='--', alpha=0.6)
    
    # 2. Тепловая карта весов внимания
    # .reshape(1, -1) делает из 1D массива 2D массив для imshow
    im = ax[1].imshow(attention_weights.reshape(1, -1), cmap='Reds', aspect='auto')
    ax[1].set_xlabel("Временные отсчеты")
    ax[1].set_ylabel("Внимание")
    ax[1].set_yticks([]) # Убираем тики по оси Y для красоты

    # Добавляем colorbar
    fig.colorbar(im, ax=ax[1], orientation='horizontal', fraction=0.05, pad=0.2)
    
    plt.tight_layout()
    plt.show()

# --- Пример использования ---
DEVICE = "cpu"
model_with_attn = HierarchicalSignalNet_WithAttention().to(DEVICE)
model_with_attn.eval() # Переводим в режим оценки

# Возьмем один сигнал
from src.dataloader.dataloader import create_dummy_data
signals, _, _, _ = create_dummy_data(1, 1, 1)
signal_np = signals[0]
signal_tensor = torch.tensor(signal_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

# Получаем выходы модели
# Нам не нужен градиент для этого
with torch.no_grad():
    embedding, logits, attn_weights = model_with_attn(signal_tensor)

# Визуализируем!
# .cpu().numpy().flatten() переводит тензор в numpy массив для matplotlib
visualize_attention(signal_np, attn_weights.cpu().numpy().flatten())
```

Теперь у вас есть не только потенциально более мощная модель, но и прямой, элегантный способ заглянуть внутрь ее "мыслей" и понять, на основании каких участков сигнала она строит свои выводы.