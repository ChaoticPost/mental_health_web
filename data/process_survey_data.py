"""
Скрипт для обработки датасета "Mental Health in Tech Survey"
и подготовки его к обучению модели определения рисков психического здоровья.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Загрузка данных
def load_survey_data(file_path):
    """
    Загружает данные из CSV-файла.
    
    Args:
        file_path (str): Путь к CSV-файлу.
        
    Returns:
        pd.DataFrame: Загруженный датафрейм.
    """
    df = pd.read_csv(file_path)
    print(f"Загружено {len(df)} записей из {file_path}")
    return df

# Предобработка данных
def preprocess_survey_data(df):
    """
    Предобрабатывает данные из опроса о психическом здоровье.
    
    Args:
        df (pd.DataFrame): Исходный датафрейм.
        
    Returns:
        pd.DataFrame: Обработанный датафрейм.
    """
    # Копируем датафрейм, чтобы не изменять оригинал
    df_processed = df.copy()
    
    # Заполняем пропущенные значения
    df_processed = df_processed.fillna("Unknown")
    
    # Создаем текстовое описание на основе ответов на опрос
    df_processed['text'] = df_processed.apply(
        lambda row: create_text_from_survey(row), axis=1
    )
    
    # Определяем уровень риска на основе ответов
    df_processed['risk_level'] = df_processed.apply(
        lambda row: determine_risk_level(row), axis=1
    )
    
    # Определяем метку класса
    df_processed['label'] = df_processed['risk_level'].map({
        'high': 'suicidal',
        'medium': 'depression',
        'low': 'normal'
    })
    
    # Выбираем только нужные колонки
    result_df = df_processed[['text', 'label', 'risk_level']]
    
    return result_df

# Создание текстового описания из ответов на опрос
def create_text_from_survey(row):
    """
    Создает текстовое описание на основе ответов на опрос.
    
    Args:
        row (pd.Series): Строка датафрейма с ответами.
        
    Returns:
        str: Текстовое описание.
    """
    text_parts = []
    
    # Добавляем информацию о лечении
    if row['treatment'] == 'Yes':
        text_parts.append("I am currently seeking treatment for mental health.")
    elif row['treatment'] == 'No':
        text_parts.append("I am not seeking any treatment for mental health.")
    
    # Добавляем информацию о влиянии на работу
    if row['work_interfere'] == 'Often':
        text_parts.append("My mental health often interferes with my work.")
    elif row['work_interfere'] == 'Sometimes':
        text_parts.append("My mental health sometimes interferes with my work.")
    elif row['work_interfere'] == 'Rarely':
        text_parts.append("My mental health rarely interferes with my work.")
    elif row['work_interfere'] == 'Never':
        text_parts.append("My mental health never interferes with my work.")
    
    # Добавляем информацию о семейной истории
    if row['family_history'] == 'Yes':
        text_parts.append("I have a family history of mental health issues.")
    elif row['family_history'] == 'No':
        text_parts.append("I don't have a family history of mental health issues.")
    
    # Добавляем информацию о поиске помощи
    if row['seek_help'] == 'Yes':
        text_parts.append("I would seek help for mental health issues.")
    elif row['seek_help'] == 'No':
        text_parts.append("I would not seek help for mental health issues.")
    elif row['seek_help'] == 'Don\'t know':
        text_parts.append("I'm not sure if I would seek help for mental health issues.")
    
    # Добавляем информацию о последствиях для психического здоровья
    if row['mental_health_consequence'] == 'Yes':
        text_parts.append("I feel there would be negative consequences for discussing mental health issues.")
    elif row['mental_health_consequence'] == 'No':
        text_parts.append("I don't think there would be negative consequences for discussing mental health issues.")
    elif row['mental_health_consequence'] == 'Maybe':
        text_parts.append("There might be negative consequences for discussing mental health issues.")
    
    # Добавляем комментарии, если они есть
    if row['comments'] != 'Unknown' and isinstance(row['comments'], str):
        text_parts.append(f"Additional thoughts: {row['comments']}")
    
    # Объединяем все части в одно текстовое описание
    return " ".join(text_parts)

# Определение уровня риска на основе ответов
def determine_risk_level(row):
    """
    Определяет уровень риска на основе ответов на опрос.
    
    Args:
        row (pd.Series): Строка датафрейма с ответами.
        
    Returns:
        str: Уровень риска ('high', 'medium', 'low').
    """
    # Факторы высокого риска
    high_risk_factors = [
        row['treatment'] == 'Yes' and row['work_interfere'] == 'Often',
        row['mental_health_consequence'] == 'Yes' and row['work_interfere'] in ['Often', 'Sometimes']
    ]
    
    # Факторы среднего риска
    medium_risk_factors = [
        row['treatment'] == 'Yes',
        row['work_interfere'] in ['Sometimes', 'Often'],
        row['family_history'] == 'Yes' and row['seek_help'] == 'No',
        row['mental_health_consequence'] == 'Maybe'
    ]
    
    # Определяем уровень риска
    if any(high_risk_factors):
        return 'medium'  # Используем 'medium' вместо 'high', так как данные не содержат явных суицидальных мыслей
    elif any(medium_risk_factors):
        return 'medium'
    else:
        return 'low'

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
    train_path = f"{output_dir}/survey_train.csv"
    test_path = f"{output_dir}/survey_test.csv"
    
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
    df = load_survey_data(input_file)
    
    # Предобработка данных
    processed_df = preprocess_survey_data(df)
    
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
    input_file = "/home/ubuntu/upload/survey.csv"
    output_dir = "/home/ubuntu/mental_health_project_package/data/processed"
    
    # Создаем директорию для сохранения, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Обработка данных
    main(input_file, output_dir)
