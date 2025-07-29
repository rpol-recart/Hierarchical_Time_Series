#Эта модель состоит из общего кодировщика и двух "голов": для эмбеддингов и 
# для поиска ключевых точек.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# Для воспроизводимости результатов
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Определяем устройство для вычислений
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {DEVICE}")

class HierarchicalSignalNet(nn.Module):
    """
    Нейросеть для иерархического кодирования сигналов и детекции ключевых точек.
    """
    def __init__(self, input_features=2, seq_len=1000, cnn_channels=64, rnn_hidden=128, embedding_dim=128, num_keypoints=2):
        super(HierarchicalSignalNet, self).__init__()
        
        self.seq_len = seq_len
        self.num_keypoints = num_keypoints
        
        # --- 1. Общий кодировщик (Encoder) ---
        
        # Сверточная часть для извлечения локальных признаков
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=cnn_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels),
            nn.MaxPool1d(kernel_size=2, stride=2), # 1000 -> 500
            
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.MaxPool1d(kernel_size=2, stride=2), # 500 -> 250
            
            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 250 -> 125
        )
        
        # Рекуррентная часть для анализа последовательности признаков
        self.encoder_rnn = nn.LSTM(
            input_size=cnn_channels * 4,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # --- 2. Голова для метрического обучения (Embedding Head) ---
        self.embedding_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden), # *2 из-за bidirectional
            nn.ReLU(),
            nn.Linear(rnn_hidden, embedding_dim)
        )
        
        # --- 3. Голова для детекции ключевых точек (Keypoint Head) ---
        self.keypoint_head = nn.Sequential(
            nn.Linear(rnn_hidden * 2, rnn_hidden),
            nn.ReLU(),
            nn.Linear(rnn_hidden, self.num_keypoints)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Входной тензор x имеет размер (batch, seq_len, features)
        
        # --- Прогон через кодировщик ---
        
        # Для Conv1d нужен формат (batch, features, seq_len)
        #x = x.permute(0, 2, 1)
        cnn_out = self.encoder_cnn(x)
        
        # Для LSTM нужен формат (batch, seq_len, features)
        cnn_out = cnn_out.permute(0, 2, 1)
        rnn_out, _ = self.encoder_rnn(cnn_out) # rnn_out: (batch, 125, rnn_hidden*2)
        
        # --- Выходы голов ---
        
        # 1. Эмбеддинг (агрегируем выходы RNN)
        # Усредняем выходы RNN по временной оси для получения одного вектора на сигнал
        aggregated_features = torch.mean(rnn_out, dim=1)
        embedding = self.embedding_head(aggregated_features)
        
        # 2. Ключевые точки
        # В отличие от эмбеддинга, здесь мы хотим получить предсказание для КАЖДОГО временного шага
        # Чтобы получить предсказания для исходной длины 1000, мы можем использовать
        # транспонированную свертку (апсемплинг) или линейный слой на выходе RNN
        # и затем интерполировать. Для простоты применим линейный слой к каждому
        # временному шагу выхода RNN, а затем сделаем апсемплинг.
        
        # Применяем полносвязный слой к каждому временному шагу
        keypoint_features = self.keypoint_head(rnn_out) # (batch, 125, num_keypoints)
        
        # Возвращаем в формат (batch, num_keypoints, 125) для апсемплинга
        keypoint_features = keypoint_features.permute(0, 2, 1)
        
        # Апсемплинг до исходной длины последовательности
        keypoint_logits = nn.functional.interpolate(
            keypoint_features, size=self.seq_len, mode='linear', align_corners=False
        ) # (batch, num_keypoints, seq_len)
        
        # Возвращаем в (batch, seq_len, num_keypoints) для удобства
        keypoint_logits = keypoint_logits.permute(0, 2, 1)
        
        return embedding, keypoint_logits
    
# объединяет TripletLoss для иерархической метрики и CrossEntropyLoss для ключевых точек.
class HierarchicalLoss(nn.Module):
    """
    Иерархическая функция потерь (ИСПРАВЛЕННАЯ ВЕРСИЯ).
    Комбинирует Triplet Loss для двух уровней и CrossEntropy для ключевых точек.
    Логика Triplet Loss реализована вручную для поддержки динамических margins.
    """
    def __init__(self, lambda_metric=1.0, margin_session=0.2, margin_element=0.5):
        super(HierarchicalLoss, self).__init__()
        self.lambda_metric = lambda_metric
        self.margin_session = margin_session
        self.margin_element = margin_element
        
        # Потери для ключевых точек. Предсказываем позицию (класс) в последовательности.
        self.keypoint_loss_fn = nn.CrossEntropyLoss()
        
        # Строка, вызывавшая ошибку, УДАЛЕНА.
        # self.triplet_loss_fn = nn.TripletMarginLoss(margin=0.0, reduction='mean') # <-- УДАЛЕНО

    def forward(self, embeddings: torch.Tensor, keypoint_logits: torch.Tensor, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        
        # --- 1. Расчет потерь для ключевых точек (L_key_points) ---
        
        logits_for_loss = keypoint_logits.permute(0, 2, 1) # -> (batch, num_keypoints, seq_len)
        
        loss_kp = 0
        target_keypoints = targets['keypoints'].to(keypoint_logits.device) # Перемещаем таргеты на нужное устройство
        for i in range(logits_for_loss.shape[1]): # Итерируемся по каждой ключевой точке
            loss_kp += self.keypoint_loss_fn(logits_for_loss[:, i, :], target_keypoints[:, i])
        
        loss_kp /= logits_for_loss.shape[1]

        # --- 2. Расчет потерь для метрического обучения (L_metric) ---
        
        anchor_indices = targets['anchor_indices']
        positive_indices = targets['positive_indices']
        session_negative_indices = targets['session_negative_indices']
        element_negative_indices = targets['element_negative_indices']

        # Проверяем, есть ли вообще триплеты в батче
        if len(anchor_indices) == 0:
            # Если триплетов нет, возвращаем только потери по ключевым точкам
            loss_components = {
                'total_loss': loss_kp.item(), 'keypoint_loss': loss_kp.item(),
                'metric_loss': 0, 'session_loss': 0, 'element_loss': 0
            }
            return loss_kp, loss_components

        anchor_emb = embeddings[anchor_indices]
        positive_emb = embeddings[positive_indices]
        session_neg_emb = embeddings[session_negative_indices]
        element_neg_emb = embeddings[element_negative_indices]
        
        # Рассчитываем евклидовы расстояния
        dist_pos = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        dist_neg_session = F.pairwise_distance(anchor_emb, session_neg_emb, p=2)
        dist_neg_element = F.pairwise_distance(anchor_emb, element_neg_emb, p=2)
        
        # Потери на уровне замера: max(0, D(a,p) - D(a,n_session) + margin_session)
        loss_session_values = torch.relu(dist_pos - dist_neg_session + self.margin_session)
        
        # Потери на уровне элемента: max(0, D(a,p) - D(a,n_element) + margin_element)
        loss_element_values = torch.relu(dist_pos - dist_neg_element + self.margin_element)

        # Усредняем потери по всем триплетам в батче
        loss_session = loss_session_values.mean()
        loss_element = loss_element_values.mean()

        loss_metric = loss_session + loss_element
        
        # --- 3. Итоговая функция потерь ---
        
        total_loss = loss_kp + self.lambda_metric * loss_metric
        
        loss_components = {
            'total_loss': total_loss.item(),
            'keypoint_loss': loss_kp.item(),
            'metric_loss': loss_metric.item(),
            'session_loss': loss_session.item(),
            'element_loss': loss_element.item()
        }
        
        return total_loss, loss_components
