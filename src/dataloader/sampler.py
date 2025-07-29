import torch
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np
import random

class HierarchicalSampler(Sampler):
    """
    Семплер, который формирует батчи по принципу P-K.
    Каждый батч содержит K сигналов для P случайно выбранных измерительных элементов.

    Args:
        dataset (Dataset): Экземпляр SignalDataset, содержащий поля element_ids.
        p (int): Количество уникальных измерительных элементов в батче.
        k (int): Количество сигналов для каждого элемента в батче.
    """
    def __init__(self, dataset, p: int, k: int):
        super().__init__(dataset)
        if not hasattr(dataset, 'element_ids'):
            raise ValueError("Датасет должен иметь атрибут 'element_ids'")
            
        self.dataset = dataset
        self.p = p
        self.k = k
        
        # 1. Создаем словарь для быстрого доступа: element_id -> [list of indices]
        self.indices_by_element = defaultdict(list)
        for idx, el_id in enumerate(self.dataset.element_ids):
            self.indices_by_element[el_id].append(idx)
            
        # 2. Список уникальных ID элементов, у которых достаточно примеров
        self.unique_element_ids = [
            el_id for el_id, indices in self.indices_by_element.items() if len(indices) >= self.k
        ]
        
        if len(self.unique_element_ids) < self.p:
            raise ValueError(f"Недостаточно элементов с >= {self.k} примерами. "
                             f"Найдено {len(self.unique_element_ids)}, требуется {self.p}.")

        # 3. Расчет длины (количества батчей за эпоху)
        self.num_batches = len(self.dataset) // (self.p * self.k)

    def __iter__(self):
        # Копия списка ID для перемешивания в каждой эпохе
        available_elements = self.unique_element_ids.copy()
        
        for _ in range(self.num_batches):
            # 1. Случайно выбираем P элементов без повторений
            random.shuffle(available_elements)
            # Если элементов меньше чем P, берем всех
            selected_elements = available_elements[:self.p]
            
            batch_indices = []
            # 2. Для каждого выбранного элемента выбираем K сигналов
            for el_id in selected_elements:
                element_indices = self.indices_by_element[el_id]
                # Выбираем K случайных индексов из доступных для этого элемента
                selected_indices = random.sample(element_indices, self.k)
                batch_indices.extend(selected_indices)
            
            # 3. Возвращаем готовый батч индексов
            yield batch_indices

    def __len__(self):
        return self.num_batches