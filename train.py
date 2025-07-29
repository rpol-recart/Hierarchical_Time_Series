import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import random
from collections import defaultdict
import argparse
import os
from tqdm import tqdm # Для наглядного прогресс-бара

# --- Импорты из вашего проекта (предполагается, что они существуют) ---
# Убедитесь, что эти файлы находятся в указанных путях
from src.dataloader.dataloader import create_dummy_data, SignalDataset, hierarchical_collate_fn
from src.model.siam import HierarchicalSignalNet, HierarchicalLoss
from src.dataloader.sampler import HierarchicalSampler

# --- Интеграция с ClearML ---
# pip install clearml
try:
    from clearml import Task
except ImportError:
    Task = None
    print("ClearML не найден. Для логирования экспериментов, установите его: pip install clearml")


def setup_clearml(args):
    """Инициализирует задачу в ClearML."""
    if Task:
        task = Task.init(
            project_name='Hierarchical Signal Training',
            task_name='Training Run',
            # Воспроизводимость: ClearML сохранит хэш коммита
            auto_connect_frameworks={'pytorch': True, 'numpy': True},
        )
        # ClearML автоматически подхватит аргументы из argparse
        # и сохранит их в разделе "Hyperparameters"
        task.connect(args, name='Конфигурация')
        print("Задача ClearML успешно инициализирована.")
        return task
    return None

def set_seed(seed_value: int):
    """Устанавливает seed для воспроизводимости."""
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        # Эти флаги могут замедлить обучение, но улучшают воспроизводимость
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch_num, num_epochs, logger):
    """Цикл обучения для одной эпохи."""
    model.train()
    epoch_losses = defaultdict(list)
    
    progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch_num+1}/{num_epochs}", leave=False)
    
    for i, batch in enumerate(progress_bar):
        # Предполагаем, что collate_fn теперь не возвращает None.
        # Если батч не может быть сформирован, это должно вызывать ошибку
        # на уровне семплера или collate_fn, что является более правильным поведением.
        if batch is None:
            print(f"Пропущен пустой батч на итерации {i}. Рекомендуется исправить семплер/collate_fn.")
            continue

        signals = batch['signals'].to(device)
        #(batch,channels,seg_len)
        signals=signals.permute(0,2,1)
        
        optimizer.zero_grad()
        
        embeddings, keypoint_logits = model(signals)
        
        loss, loss_components = loss_fn(embeddings, keypoint_logits, batch)
        
        if torch.isnan(loss):
            print(f"Внимание: получено значение NaN для функции потерь на итерации {i}. Пропускаем шаг.")
            continue
            
        loss.backward()
        # Опционально: обрезка градиентов для предотвращения "взрыва"
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Логирование метрик на каждой итерации
        global_step = epoch_num * len(dataloader) + i
        for key, value in loss_components.items():
                epoch_losses[key].append(value)
        if logger:
            logger.report_scalar(title="Loss per Iteration", series="Total Loss", value=loss.item(), iteration=global_step)
            for key, value in loss_components.items():
                epoch_losses[key].append(value)
                logger.report_scalar(title="Loss Components per Iteration", series=key, value=value, iteration=global_step)

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'kp_loss': f"{loss_components.get('keypoint_loss', 0):.4f}",
            'metric_loss': f"{loss_components.get('metric_loss', 0):.4f}"
        })
        
    return epoch_losses


def run_training(args):
    """Основная функция, запускающая процесс обучения."""
    
    # 0. Инициализация ClearML и установка seed
    logger = setup_clearml(args).get_logger() if Task else None
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # 1. Создание данных и датасета
    print("Создание dummy-данных...")
    signals, element_ids, session_ids, keypoints = create_dummy_data(
        num_elements=args.num_elements, 
        sessions_per_element=args.sessions_per_element, 
        signals_per_session=args.signals_per_session
    )
    dataset = SignalDataset(signals, element_ids, session_ids, keypoints)
    
    # Здесь можно было бы разделить на train/validation, но для иерархического семплера
    # это сложнее. Оставим один датасет, как в оригинале.
    
    # 2. Создание семплера и DataLoader
    print(f"Создание семплера с P={args.p_elements}, K={args.k_samples}.")
    sampler = HierarchicalSampler(dataset, p=args.p_elements, k=args.k_samples)
    
    # Размер батча определяется семплером: P * K
    batch_size = args.p_elements * args.k_samples
    print(f"Размер батча: {batch_size}")
    
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler, # <--- ИСПОЛЬЗУЕМ batch_sampler
        # При использовании batch_sampler, аргументы batch_size, shuffle, sampler и drop_last должны быть None (по умолчанию)
        collate_fn=hierarchical_collate_fn,
        num_workers=2, # num_workers можно оставить
        pin_memory=True # Ускоряет передачу данных на GPU
    )
   

    # 3. Инициализация модели, функции потерь и оптимизатора
    print(f"Инициализация модели с размерностью эмбеддинга: {args.embedding_dim}")
    model = HierarchicalSignalNet(embedding_dim=args.embedding_dim).to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/model_epoch_20.pth")) 
    except FileNotFoundError:
        print("Внимание: Файл с весами модели не найден. Будут использованы случайные веса.")
    loss_fn = HierarchicalLoss(
        lambda_metric=args.lambda_metric, 
        margin_session=args.margin_session, 
        margin_element=args.margin_element
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Опционально: планировщик скорости обучения
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 4. Главный цикл обучения
    print("\nНачало обучения...")
    for epoch in range(args.num_epochs):
        epoch_losses = train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch, args.num_epochs, logger)
        
        # Логирование средних потерь за эпоху
        avg_total_loss = np.mean(epoch_losses['total_loss']) if epoch_losses['total_loss'] else 0
        avg_kp_loss = np.mean(epoch_losses['keypoint_loss']) if epoch_losses['keypoint_loss'] else 0
        avg_metric_loss = np.mean(epoch_losses['metric_loss']) if epoch_losses['metric_loss'] else 0
        
        log_message = (f"Итоги эпохи {epoch+1}/{args.num_epochs} | "
                       f"Общая ошибка: {avg_total_loss:.4f} | "
                       f"Ошибка KP: {avg_kp_loss:.4f} | "
                       f"Ошибка метрики: {avg_metric_loss:.4f}")
        print(log_message)
        
        if logger:
            logger.report_scalar(title="Average Epoch Loss", series="Total Loss", value=avg_total_loss, iteration=epoch)
            logger.report_scalar(title="Average Epoch Loss", series="Keypoint Loss", value=avg_kp_loss, iteration=epoch)
            logger.report_scalar(title="Average Epoch Loss", series="Metric Loss", value=avg_metric_loss, iteration=epoch)
            current_lr = optimizer.param_groups[0]['lr']
            logger.report_scalar(title="Learning Rate", series="LR", value=current_lr, iteration=epoch)

        # Сохранение модели (чекпойнт)
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Модель сохранена в: {model_path}")
            if Task and logger:
                # Загружаем модель как артефакт в ClearML
                logger.report_artifact(name=f"model_epoch_{epoch+1}", artifact_object=model_path)
    
    print("Обучение завершено.")
    if Task and logger:
        print("Задача ClearML завершена. Результаты доступны в веб-интерфейсе.")


def main():
    parser = argparse.ArgumentParser(description="Обучение иерархической сиамской сети.")
    
    # Аргументы для семплера и данных
    parser.add_argument('--p_elements', type=int, default=8, help='P: количество элементов в батче')
    parser.add_argument('--k_samples', type=int, default=8, help='K: количество сигналов на элемент')
    parser.add_argument('--num_elements', type=int, default=20, help='Общее количество уникальных элементов')
    parser.add_argument('--sessions_per_element', type=int, default=20, help='Количество сессий на элемент')
    parser.add_argument('--signals_per_session', type=int, default=10, help='Количество сигналов на сессию')

    # Аргументы для обучения
    parser.add_argument('--num_epochs', type=int, default=20, help='Количество эпох обучения')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Скорость обучения')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Размерность пространства эмбеддингов')
    parser.add_argument('--num_workers', type=int, default=2, help='Количество воркеров для DataLoader')
    
    # Аргументы для функции потерь
    parser.add_argument('--lambda_metric', type=float, default=0.5, help='Коэффициент для metric loss')
    parser.add_argument('--margin_session', type=float, default=0.2, help='Margin для триплетов внутри сессии')
    parser.add_argument('--margin_element', type=float, default=0.5, help='Margin для триплетов между сессиями')

    # Системные аргументы
    parser.add_argument('--seed', type=int, default=42, help='Seed для воспроизводимости')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Директория для сохранения моделей')
    parser.add_argument('--save_every', type=int, default=5, help='Сохранять модель каждые N эпох')

    args = parser.parse_args()
    
    run_training(args)

if __name__ == '__main__':
    main()