"""
Скрипт для объединения обработанных датасетов и создания итоговых обучающих и тестовых CSV-файлов
для обучения гибридной модели мониторинга психического здоровья.
"""

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Создаем директорию для сохранения итоговых датасетов
os.makedirs('/home/ubuntu/mental_health_project_package/data/final', exist_ok=True)

# Загружаем обработанные датасеты
survey_train = pd.read_csv('/home/ubuntu/mental_health_project_package/data/processed/survey_train.csv')
survey_test = pd.read_csv('/home/ubuntu/mental_health_project_package/data/processed/survey_test.csv')
suicide_train = pd.read_csv('/home/ubuntu/mental_health_project_package/data/processed/suicide_train.csv')
suicide_test = pd.read_csv('/home/ubuntu/mental_health_project_package/data/processed/suicide_test.csv')

# Выводим информацию о загруженных датасетах
print("Загруженные датасеты:")
print(f"Survey Train: {len(survey_train)} записей")
print(f"Survey Test: {len(survey_test)} записей")
print(f"Suicide Train: {len(suicide_train)} записей")
print(f"Suicide Test: {len(suicide_test)} записей")

# Балансируем выборки для предотвращения перекоса в сторону суицидальных текстов
# Берем все записи из survey и случайную выборку из suicide
suicide_train_balanced = suicide_train.sample(n=min(len(suicide_train), len(survey_train)*2), random_state=42)
suicide_test_balanced = suicide_test.sample(n=min(len(suicide_test), len(survey_test)*2), random_state=42)

print(f"\nБалансированные датасеты:")
print(f"Suicide Train (сбалансированный): {len(suicide_train_balanced)} записей")
print(f"Suicide Test (сбалансированный): {len(suicide_test_balanced)} записей")

# Объединяем обучающие выборки
train_df = pd.concat([survey_train, suicide_train_balanced], ignore_index=True)

# Объединяем тестовые выборки
test_df = pd.concat([survey_test, suicide_test_balanced], ignore_index=True)

# Перемешиваем данные
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Выводим статистику по объединенным датасетам
print("\nОбъединенные датасеты:")
print(f"Train: {len(train_df)} записей")
print(f"Test: {len(test_df)} записей")

print("\nРаспределение классов в обучающей выборке:")
print(train_df['label'].value_counts())

print("\nРаспределение уровней риска в обучающей выборке:")
print(train_df['risk_level'].value_counts())

print("\nРаспределение классов в тестовой выборке:")
print(test_df['label'].value_counts())

print("\nРаспределение уровней риска в тестовой выборке:")
print(test_df['risk_level'].value_counts())

# Сохраняем итоговые датасеты
train_df.to_csv('/home/ubuntu/mental_health_project_package/data/final/train.csv', index=False)
test_df.to_csv('/home/ubuntu/mental_health_project_package/data/final/test.csv', index=False)

print("\nИтоговые датасеты успешно созданы и сохранены в директории /home/ubuntu/mental_health_project_package/data/final/")

# Создаем примеры для демонстрации
examples = [
    "I want to kill myself, I can't take it anymore.",
    "I no longer see the point of living.",
    "I feel so depressed lately, nothing brings me joy.",
    "I'm struggling to get out of bed every morning.",
    "I had a bad day today, but tomorrow will be better.",
    "Work has been stressful, but I'm managing."
]

examples_df = pd.DataFrame({
    'text': examples,
    'expected_risk': ['high', 'high', 'medium', 'medium', 'low', 'low']
})

examples_df.to_csv('/home/ubuntu/mental_health_project_package/data/final/examples.csv', index=False)

print("\nФайл с примерами для тестирования создан: /home/ubuntu/mental_health_project_package/data/final/examples.csv")

# Создаем README файл с описанием датасетов
readme_content = """# Датасеты для обучения модели мониторинга психического здоровья

## Описание файлов

- `train.csv` - Обучающая выборка, объединяющая данные из опроса о психическом здоровье и суицидальные тексты
- `test.csv` - Тестовая выборка для оценки качества модели
- `examples.csv` - Примеры текстов для тестирования модели

## Структура данных

Каждый файл содержит следующие колонки:
- `text` - Текстовое сообщение для анализа
- `label` - Метка класса ('suicidal', 'depression', 'normal')
- `risk_level` - Уровень риска ('high', 'medium', 'low')

## Источники данных

1. **Mental Health in Tech Survey** - Опрос о психическом здоровье в технологической индустрии
2. **Suicide and Depression Detection** - Датасет с суицидальными и несуицидальными текстами

## Статистика

### Обучающая выборка
- Всего записей: {train_count}
- Распределение классов:
  - suicidal: {train_suicidal}
  - depression: {train_depression}
  - normal: {train_normal}

### Тестовая выборка
- Всего записей: {test_count}
- Распределение классов:
  - suicidal: {test_suicidal}
  - depression: {test_depression}
  - normal: {test_normal}

## Использование

Эти датасеты предназначены для обучения и тестирования модели определения рисков психического здоровья по текстовым сообщениям.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Загрузка данных
train_df = pd.read_csv('data/final/train.csv')
test_df = pd.read_csv('data/final/test.csv')

# Подготовка данных
X_train = train_df['text']
y_train = train_df['risk_level']
X_test = test_df['text']
y_test = test_df['risk_level']

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

# Оценка качества
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
```
""".format(
    train_count=len(train_df),
    train_suicidal=len(train_df[train_df['label'] == 'suicidal']),
    train_depression=len(train_df[train_df['label'] == 'depression']),
    train_normal=len(train_df[train_df['label'] == 'normal']),
    test_count=len(test_df),
    test_suicidal=len(test_df[test_df['label'] == 'suicidal']),
    test_depression=len(test_df[test_df['label'] == 'depression']),
    test_normal=len(test_df[test_df['label'] == 'normal'])
)

with open('/home/ubuntu/mental_health_project_package/data/final/README.md', 'w') as f:
    f.write(readme_content)

print("\nREADME файл с описанием датасетов создан: /home/ubuntu/mental_health_project_package/data/final/README.md")
