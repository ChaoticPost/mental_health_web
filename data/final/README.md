# Датасеты для обучения модели мониторинга психического здоровья

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
- Всего записей: 3021
- Распределение классов:
  - suicidal: 1010
  - depression: 769
  - normal: 1242

### Тестовая выборка
- Всего записей: 756
- Распределение классов:
  - suicidal: 260
  - depression: 192
  - normal: 304

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
