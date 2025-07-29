import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple

def create_dummy_data(num_elements=10, sessions_per_element=20, signals_per_session=8, seq_len=1000, features=2):
    """Генерирует фиктивные данные, имитирующие реальную структуру."""
    print("Генерация фиктивных данных...")
    signals, element_ids, session_ids, keypoints = [], [], [], []
    total_signals = num_elements * sessions_per_element * signals_per_session
    
    current_session_id = 0
    for el_id in range(num_elements):
        for ses_id_local in range(sessions_per_element):
            for _ in range(signals_per_session):
                # Создаем сигнал (просто шум)
                signal = np.random.randn(seq_len, features).astype(np.float32)
                # Добавляем "особенность" для элемента и замера
                signal += np.sin(np.linspace(0, (el_id + 1) * np.pi, seq_len))[:, None] * 0.5
                signal += np.random.randn() * 0.2 # шум замера
                
                signals.append(signal)
                element_ids.append(el_id)
                session_ids.append(current_session_id)
                
                # Генерируем 2 случайные ключевые точки
                kp = sorted(np.random.randint(0, seq_len, size=2))
                keypoints.append(np.array(kp, dtype=np.int64))

            current_session_id += 1
            
    print(f"Сгенерировано {len(signals)} сигналов.")
    return signals, np.array(element_ids), np.array(session_ids), keypoints

class SignalDataset(Dataset):
    """Простой датасет для хранения сигналов и их метаданных."""
    def __init__(self, signals, element_ids, session_ids, keypoints):
        self.signals = signals
        self.element_ids = element_ids
        self.session_ids = session_ids
        self.keypoints = keypoints

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return {
            "signal": self.signals[idx],
            "element_id": self.element_ids[idx],
            "session_id": self.session_ids[idx],
            "keypoints": self.keypoints[idx]
        }

def hierarchical_collate_fn(batch: List[Dict]) -> Dict:
    """
    Специальная функция для сборки батча и генерации триплетов "на лету".
    """
    signals = torch.stack([torch.from_numpy(item['signal']) for item in batch])
    keypoints = torch.stack([torch.from_numpy(item['keypoints']) for item in batch])
    element_ids = np.array([item['element_id'] for item in batch])
    session_ids = np.array([item['session_id'] for item in batch])
    
    anchor_indices, positive_indices = [], []
    session_negative_indices, element_negative_indices = [], []
    
    # Группируем индексы батча по метаданным для быстрого поиска
    indices_by_session = defaultdict(list)
    indices_by_element = defaultdict(list)
    for i in range(len(batch)):
        indices_by_session[session_ids[i]].append(i)
        indices_by_element[element_ids[i]].append(i)
        
    for i in range(len(batch)):
        # i - наш якорь (anchor)
        anchor_el_id = element_ids[i]
        anchor_ses_id = session_ids[i]
        
        # 1. Найти позитивный пример (тот же замер, другой сигнал)
        pos_candidates = [p for p in indices_by_session[anchor_ses_id] if p != i]
        if not pos_candidates: continue # Пропускаем, если в батче нет пары
        
        # 2. Найти негатив на уровне замера (тот же элемент, другой замер)
        ses_neg_candidates = []
        for ses_idx in indices_by_element[anchor_el_id]:
            if session_ids[ses_idx] != anchor_ses_id:
                ses_neg_candidates.append(ses_idx)
        if not ses_neg_candidates: continue # Пропускаем
                
        # 3. Найти негатив на уровне элемента (другой элемент)
        el_neg_candidates = [n for n in range(len(batch)) if element_ids[n] != anchor_el_id]
        if not el_neg_candidates: continue # Пропускаем
        
        # Если все кандидаты найдены, выбираем случайных и добавляем в списки
        anchor_indices.append(i)
        positive_indices.append(random.choice(pos_candidates))
        session_negative_indices.append(random.choice(ses_neg_candidates))
        element_negative_indices.append(random.choice(el_neg_candidates))

    # ВАЖНО: Если в батче не удалось сформировать триплеты, списки индексов будут пустыми.
    # В реальном приложении лучше использовать специальный семплер (Sampler),
    # который гарантирует наличие нужных примеров в батче. Для демонстрации
    # этот метод достаточен. Мы просто вернем пустые тензоры, если ничего не нашлось.
    if not anchor_indices:
        return None

    return {
        "signals": signals,
        "keypoints": keypoints,
        "element_ids": torch.from_numpy(element_ids),
        "session_ids": torch.from_numpy(session_ids),
        "anchor_indices": torch.tensor(anchor_indices, dtype=torch.long),
        "positive_indices": torch.tensor(positive_indices, dtype=torch.long),
        "session_negative_indices": torch.tensor(session_negative_indices, dtype=torch.long),
        "element_negative_indices": torch.tensor(element_negative_indices, dtype=torch.long)
    }