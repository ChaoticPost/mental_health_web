"""
Улучшенный скрипт для обучения ML-модели на объединенных датасетах
и интеграции с эвристическим анализатором в гибридную систему.
"""

import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Создаем директории для сохранения моделей и конфигураций
os.makedirs('src/models/ml', exist_ok=True)
os.makedirs('src/models/config', exist_ok=True)

# Функция для загрузки данных
def load_data(train_path, test_path):
    """
    Загружает обучающую и тестовую выборки.
    
    Args:
        train_path (str): Путь к файлу с обучающей выборкой.
        test_path (str): Путь к файлу с тестовой выборкой.
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - данные для обучения и тестирования.
    """
    # Загрузка данных
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Подготовка данных
    X_train = train_df['text']
    y_train = train_df['risk_level']
    X_test = test_df['text']
    y_test = test_df['risk_level']
    
    print(f"Загружено {len(X_train)} обучающих примеров и {len(X_test)} тестовых примеров")
    print(f"Распределение классов в обучающей выборке: {y_train.value_counts().to_dict()}")
    print(f"Распределение классов в тестовой выборке: {y_test.value_counts().to_dict()}")
    
    return X_train, y_train, X_test, y_test

# Функция для предобработки текста
def preprocess_text(text):
    """
    Предобрабатывает текст для анализа.
    
    Args:
        text (str): Исходный текст.
        
    Returns:
        str: Предобработанный текст.
    """
    if not isinstance(text, str):
        return ""
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Токенизация
    tokens = nltk.word_tokenize(text)
    
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Объединение токенов обратно в текст
    processed_text = ' '.join(tokens)
    
    return processed_text

# Функция для обучения модели
def train_model(X_train, y_train):
    """
    Обучает модель на предоставленных данных с оптимизацией гиперпараметров.
    
    Args:
        X_train (pd.Series): Тексты для обучения.
        y_train (pd.Series): Метки классов для обучения.
        
    Returns:
        Pipeline: Обученная модель.
    """
    # Создаем пайплайн
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(preprocessor=preprocess_text)),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Определяем параметры для оптимизации
    param_grid = {
        'vectorizer__max_features': [3000, 5000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20]
    }
    
    # Оптимизируем гиперпараметры
    print("Начинаем оптимизацию гиперпараметров...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Выводим лучшие параметры
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучший результат: {grid_search.best_score_:.4f}")
    
    # Возвращаем лучшую модель
    return grid_search.best_estimator_

# Функция для оценки модели
def evaluate_model(model, X_test, y_test):
    """
    Оценивает качество модели на тестовой выборке.
    
    Args:
        model (Pipeline): Обученная модель.
        X_test (pd.Series): Тексты для тестирования.
        y_test (pd.Series): Метки классов для тестирования.
        
    Returns:
        dict: Метрики качества модели.
    """
    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Выводим метрики
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Выводим полный отчет
    print("\nКлассификационный отчет:")
    print(classification_report(y_test, y_pred))
    
    # Возвращаем метрики
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

# Функция для сохранения модели
def save_model(model, model_path):
    """
    Сохраняет модель в файл.
    
    Args:
        model (Pipeline): Обученная модель.
        model_path (str): Путь для сохранения модели.
        
    Returns:
        str: Путь к сохраненной модели.
    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Модель сохранена в {model_path}")
    
    return model_path

# Функция для создания конфигурации эвристического анализатора
def create_heuristic_config(config_path):
    """
    Создает конфигурацию для эвристического анализатора.
    
    Args:
        config_path (str): Путь для сохранения конфигурации.
        
    Returns:
        str: Путь к сохраненной конфигурации.
    """
    # Определяем конфигурацию
    config = {
        "high_risk_patterns": [
            r"(?i)(?:want|going|plan(?:ning)?|think(?:ing)?|consider(?:ing)?)\s+(?:to|about)\s+(?:kill(?:ing)?|end(?:ing)?|hurt(?:ing)?|harm(?:ing)?)\s+(?:myself|my\s+life|me)",
            r"(?i)(?:no|don't)\s+(?:longer|anymore|want|wish|desire)\s+(?:to|for)\s+(?:live|living|be\s+alive|exist|wake\s+up)",
            r"(?i)(?:suicid(?:e|al)|kill(?:ing)?\s+myself|end(?:ing)?\s+my\s+life|take\s+my\s+own\s+life)",
            r"(?i)(?:better\s+off\s+(?:without\s+me|dead)|nobody\s+(?:would\s+miss\s+me|cares\s+about\s+me))",
            r"(?i)(?:written|writing|prepared|preparing)\s+(?:my|a)\s+(?:suicide\s+note|goodbye\s+letter)",
            r"(?i)(?:jump(?:ing)?|leap(?:ing)?)\s+(?:off|from)\s+(?:a|the)\s+(?:bridge|building|roof|window)",
            r"(?i)(?:hang(?:ing)?|shoot(?:ing)?|stab(?:bing)?|cut(?:ting)?)\s+myself",
            r"(?i)(?:stockpil(?:e|ing)|collect(?:ing)?|sav(?:e|ing))\s+(?:pills|medication|drugs)",
            r"(?i)(?:overdos(?:e|ing)|poison(?:ing)?)\s+(?:myself|me)",
            r"(?i)(?:can't|cannot|don't\s+want\s+to)\s+(?:go\s+on|continue|keep\s+going|live\s+like\s+this)",
            r"(?i)(?:life\s+(?:is|has\s+become)\s+(?:unbearable|pointless|meaningless|hopeless))",
            r"(?i)(?:no\s+(?:reason|point)\s+(?:to|in|for)\s+(?:live|living|continue|go(?:ing)?\s+on))",
            r"(?i)(?:burden\s+to\s+(?:everyone|my\s+family|others|society))",
            r"(?i)(?:never\s+wake\s+up|sleep\s+forever|permanent\s+solution)",
            r"(?i)(?:set\s+a\s+date|made\s+a\s+plan|decided\s+when|chosen\s+a\s+method)\s+(?:for|to)\s+(?:suicide|kill(?:ing)?\s+myself|end(?:ing)?\s+my\s+life)"
        ],
        "medium_risk_patterns": [
            r"(?i)(?:feel(?:ing)?|am)\s+(?:depress(?:ed|ion)|sad|down|low|blue)",
            r"(?i)(?:no|lost|lack\s+of)\s+(?:interest|motivation|energy|joy|pleasure)",
            r"(?i)(?:struggle|hard|difficult)\s+(?:to|with)\s+(?:get(?:ting)?\s+out\s+of\s+bed|sleep(?:ing)?|eat(?:ing)?)",
            r"(?i)(?:feel(?:ing)?|am)\s+(?:worthless|hopeless|helpless|empty|numb)",
            r"(?i)(?:can't|don't|unable\s+to)\s+(?:focus|concentrate|think\s+clearly)",
            r"(?i)(?:everything|life)\s+(?:feels|seems|is)\s+(?:pointless|meaningless|overwhelming)",
            r"(?i)(?:tired|exhausted)\s+(?:all\s+the\s+time|constantly|always)",
            r"(?i)(?:cry(?:ing)?|tear(?:s|ful))\s+(?:all\s+the\s+time|constantly|frequently|often)",
            r"(?i)(?:isolat(?:e|ing)|withdraw(?:n|ing)|avoid(?:ing)?)\s+(?:myself|people|friends|family)",
            r"(?i)(?:hate|dislike|loathe)\s+(?:myself|my\s+life|who\s+I\s+am)",
            r"(?i)(?:feel(?:ing)?|am)\s+(?:trapped|stuck|lost|alone|lonely)",
            r"(?i)(?:anxiety|anxious|panic\s+attack|worry(?:ing)?)\s+(?:constant(?:ly)?|all\s+the\s+time|overwhelming)"
        ],
        "high_risk_keywords": [
            "suicide", "suicidal", "kill myself", "end my life", "take my life", 
            "want to die", "don't want to live", "better off dead", "no reason to live",
            "no point in living", "can't go on", "ending it all", "final solution",
            "goodbye letter", "suicide note", "last wish", "overdose", "hanging",
            "jump off", "slit wrists", "shoot myself", "pills", "no longer see the point",
            "no longer want to exist", "rather be dead", "wish I was dead", "wish I wasn't alive"
        ],
        "medium_risk_keywords": [
            "depressed", "depression", "hopeless", "worthless", "empty", "numb",
            "can't feel anything", "no interest", "no motivation", "no energy",
            "tired all the time", "can't sleep", "sleeping too much", "no appetite",
            "eating too much", "can't focus", "can't concentrate", "isolating myself",
            "avoiding people", "don't enjoy", "no pleasure", "hate myself", "self-loathing",
            "failure", "giving up", "overwhelmed", "can't cope", "trapped", "stuck",
            "no way out", "no future", "pointless", "meaningless", "alone", "lonely",
            "abandoned", "rejected", "unloved", "unwanted", "burden", "useless"
        ],
        "negation_terms": [
            "not", "don't", "doesn't", "didn't", "haven't", "hasn't", "hadn't",
            "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "never"
        ],
        "intensifiers": [
            "very", "really", "extremely", "completely", "totally", "absolutely",
            "utterly", "entirely", "deeply", "severely", "seriously", "terribly",
            "incredibly", "overwhelmingly", "exceedingly", "immensely", "intensely"
        ],
        "thresholds": {
            "high_risk": 0.8,
            "medium_risk": 0.6
        }
    }
    
    # Сохраняем конфигурацию
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Конфигурация эвристического анализатора сохранена в {config_path}")
    
    return config_path

# Основная функция
def main():
    """
    Основная функция для обучения и оценки модели.
    """
    # Пути к файлам
    train_path = 'data/final/train.csv'
    test_path = 'data/final/test.csv'
    model_path = 'src/models/ml/mental_health_model.pkl'
    config_path = 'src/models/config/heuristic_config.json'
    
    # Загрузка данных
    X_train, y_train, X_test, y_test = load_data(train_path, test_path)
    
    # Обучение модели
    model = train_model(X_train, y_train)
    
    # Оценка модели
    metrics = evaluate_model(model, X_test, y_test)
    
    # Сохранение модели
    save_model(model, model_path)
    
    # Создание конфигурации эвристического анализатора
    create_heuristic_config(config_path)
    
    # Сохранение метрик
    metrics_path = 'src/models/ml/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Метрики сохранены в {metrics_path}")
    
    print("\nОбучение и оценка модели завершены успешно!")
    print(f"Модель сохранена в {model_path}")
    print(f"Конфигурация эвристического анализатора сохранена в {config_path}")
    print(f"Метрики сохранены в {metrics_path}")
    
    return model, metrics

if __name__ == "__main__":
    main()
