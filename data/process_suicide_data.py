"""
Скрипт для обработки датасета Suicide_Detection.csv
и подготовки его к обучению модели определения рисков психического здоровья.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Загрузка данных
def load_suicide_data(file_path):
    """
    Загружает данные из CSV-файла с суицидальными текстами.
    
    Args:
        file_path (str): Путь к CSV-файлу.
        
    Returns:
        pd.DataFrame: Загруженный датафрейм.
    """
    df = pd.read_csv(file_path)
    print(f"Загружено {len(df)} записей из {file_path}")
    return df

# Предобработка данных
def preprocess_suicide_data(df):
    """
    Предобрабатывает данные из датасета суицидальных текстов.
    
    Args:
        df (pd.DataFrame): Исходный датафрейм.
        
    Returns:
        pd.DataFrame: Обработанный датафрейм.
    """
    # Копируем датафрейм, чтобы не изменять оригинал
    df_processed = df.copy()
    
    # Удаляем строки с пустыми текстами
    df_processed = df_processed.dropna(subset=['text'])
    
    # Преобразуем метки классов
    df_processed['label'] = df_processed['class'].map({
        'suicide': 'suicidal',
        'non-suicide': 'normal'
    })
    
    # Определяем уровень риска на основе меток
    df_processed['risk_level'] = df_processed['label'].map({
        'suicidal': 'high',
        'normal': 'low'
    })
    
    # Выбираем только нужные колонки
    result_df = df_processed[['text', 'label', 'risk_level']]
    
    return result_df

# Разделение на обучающую и тестовую выборки
def split_data(df, test_size=0.2, random_state=42):
    """
    Разделяет данные на обучающую и тестовую выборки.
    
    Args:
        df (pd.DataFrame): Исходный датафрейм.
        test_size (float): Доля тестовой выборки.
        random_state (int): Seed для воспроизводимости.
        
    Returns:
        tuple: (train_df, test_df) - обучающая и тестовая выборки.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['label']
    )
    
    print(f"Обучающая выборка: {len(train_df)} записей")
    print(f"Тестовая выборка: {len(test_df)} записей")
    
    return train_df, test_df

# Сохранение данных
def save_data(train_df, test_df, output_dir):
    """
    Сохраняет обучающую и тестовую выборки в CSV-файлы.
    
    Args:
        train_df (pd.DataFrame): Обучающая выборка.
        test_df (pd.DataFrame): Тестовая выборка.
        output_dir (str): Директория для сохранения.
        
    Returns:
        tuple: (train_path, test_path) - пути к сохраненным файлам.
    """
    train_path = f"{output_dir}/suicide_train.csv"
    test_path = f"{output_dir}/suicide_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Обучающая выборка сохранена в {train_path}")
    print(f"Тестовая выборка сохранена в {test_path}")
    
    return train_path, test_path

# Основная функция
def main(input_file, output_dir):
    """
    Основная функция для обработки данных.
    
    Args:
        input_file (str): Путь к входному CSV-файлу.
        output_dir (str): Директория для сохранения результатов.
        
    Returns:
        tuple: (train_path, test_path) - пути к сохраненным файлам.
    """
    # Загрузка данных
    df = load_suicide_data(input_file)
    
    # Предобработка данных
    processed_df = preprocess_suicide_data(df)
    
    # Разделение на обучающую и тестовую выборки
    train_df, test_df = split_data(processed_df)
    
    # Сохранение данных
    train_path, test_path = save_data(train_df, test_df, output_dir)
    
    # Вывод статистики
    print("\nРаспределение классов в обучающей выборке:")
    print(train_df['label'].value_counts())
    
    print("\nРаспределение уровней риска в обучающей выборке:")
    print(train_df['risk_level'].value_counts())
    
    return train_path, test_path

if __name__ == "__main__":
    import os
    
    # Пути к файлам
    input_file = "/home/ubuntu/mental_health_project_package/data/raw/Suicide_Detection.csv"
    output_dir = "/home/ubuntu/mental_health_project_package/data/processed"
    
    # Создаем директорию для сохранения, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Обработка данных
    main(input_file, output_dir)
