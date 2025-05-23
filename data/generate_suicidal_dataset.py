"""
Скрипт для создания синтетического датасета суицидальных текстов
для дополнения датасета "Mental Health in Tech Survey".
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Создаем директорию для сохранения датасета
os.makedirs('/home/ubuntu/mental_health_project_package/data/processed', exist_ok=True)

# Примеры текстов с суицидальными мыслями (высокий риск)
suicidal_texts = [
    "I want to kill myself, I can't take it anymore.",
    "I'm planning to end my life tonight, I've written my note.",
    "I don't see any reason to continue living, everything is pointless.",
    "I no longer see the point of living.",
    "I've been thinking about suicide a lot lately.",
    "I just want this pain to stop forever.",
    "Nobody would miss me if I was gone.",
    "I feel like I'm a burden to everyone around me.",
    "I've been giving away my things lately, won't need them soon.",
    "I've tried everything, nothing helps, I want to die.",
    "I just want to sleep and never wake up.",
    "I have a plan to end my life.",
    "I'm tired of waking up every day to the same pain.",
    "Life has no meaning for me anymore.",
    "I've been researching ways to kill myself.",
    "I'm going to jump off the bridge tonight.",
    "I've been stockpiling pills for when I'm ready.",
    "I wish I could just disappear forever.",
    "I'm a failure and everyone would be better off without me.",
    "I can't bear this pain any longer, I need it to end.",
    "I've written goodbye letters to everyone I care about.",
    "I'm sorry for what I'm about to do, but I can't go on.",
    "I've set a date for my suicide.",
    "I'm going to shoot myself this weekend.",
    "I've been cutting myself deeper each time.",
    "I'm worthless and don't deserve to live.",
    "I'm going to hang myself tonight.",
    "I've made peace with my decision to end it all.",
    "I'm a burden to my family and they'll be better without me.",
    "I've tried therapy and medication, nothing works, I'm done."
]

# Расширяем датасет с вариациями
def expand_dataset(texts, n_variations=3):
    expanded = []
    for text in texts:
        expanded.append(text)
        words = text.split()
        for _ in range(n_variations):
            if len(words) > 5:
                # Перемешиваем порядок слов или заменяем некоторые слова синонимами
                variation = ' '.join(np.random.choice(words, len(words), replace=True))
                expanded.append(variation)
    return expanded

# Расширяем наборы текстов
suicidal_expanded = expand_dataset(suicidal_texts)

# Создаем DataFrame
data = []

# Добавляем суицидальные тексты (высокий риск)
for text in suicidal_expanded:
    data.append({
        'text': text,
        'label': 'suicidal',
        'risk_level': 'high'
    })

# Создаем DataFrame
df = pd.DataFrame(data)

# Перемешиваем данные
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Разделяем на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Сохраняем датасеты
train_df.to_csv('/home/ubuntu/mental_health_project_package/data/processed/suicidal_train.csv', index=False)
test_df.to_csv('/home/ubuntu/mental_health_project_package/data/processed/suicidal_test.csv', index=False)

# Выводим статистику
print(f"Всего примеров: {len(df)}")
print(f"Обучающая выборка: {len(train_df)}")
print(f"Тестовая выборка: {len(test_df)}")
print("\nРаспределение классов в обучающей выборке:")
print(train_df['label'].value_counts())
print("\nРаспределение уровней риска в обучающей выборке:")
print(train_df['risk_level'].value_counts())

print("\nДатасет суицидальных текстов успешно создан и сохранен в директории /home/ubuntu/mental_health_project_package/data/processed/")
