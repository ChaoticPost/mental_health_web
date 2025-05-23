"""
Скрипт для создания синтетического датасета для обучения ML-модели
определения рисков психического здоровья.

Датасет имитирует структуру "Suicide and Depression Detection" и 
"Mental Health in Tech Survey" с Kaggle.
"""

import pandas as pd
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split

# Создаем директорию для сохранения датасета
os.makedirs('/home/ubuntu/mental_health_project/data/processed', exist_ok=True)

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

# Примеры текстов с депрессивными мыслями (средний риск)
depressive_texts = [
    "I feel so depressed lately, nothing brings me joy.",
    "I'm struggling to get out of bed every morning.",
    "Everything feels pointless now.",
    "I've lost interest in all the things I used to enjoy.",
    "I feel empty inside all the time.",
    "I'm so tired no matter how much I sleep.",
    "I can't focus on anything anymore.",
    "I feel like I'm drowning in sadness.",
    "I cry for no reason several times a day.",
    "I feel like a failure at everything I do.",
    "I've been isolating myself from friends and family.",
    "I don't have the energy to do basic tasks anymore.",
    "Food has lost all taste to me.",
    "I feel numb to everything around me.",
    "I can't remember the last time I felt happy.",
    "I'm constantly exhausted both mentally and physically.",
    "I feel like I'm just going through the motions of life.",
    "I've gained/lost a lot of weight recently without trying.",
    "I feel worthless compared to everyone around me.",
    "I can't seem to enjoy anything anymore.",
    "I'm always irritable and snap at people for no reason.",
    "I feel like I'm disappointing everyone in my life.",
    "I've been having trouble sleeping for weeks.",
    "I feel like I'm trapped in a dark hole with no way out.",
    "I can't stop thinking negative thoughts.",
    "I feel like a burden to everyone around me.",
    "I'm struggling with anxiety and panic attacks.",
    "I feel overwhelmed by even small tasks.",
    "I've lost all motivation to do anything.",
    "I feel like I'm watching my life from outside my body."
]

# Примеры текстов с низким риском
normal_texts = [
    "I had a bad day today, but tomorrow will be better.",
    "Work has been stressful, but I'm managing.",
    "I'm feeling a bit sad after watching that movie.",
    "I'm tired after a long day at work.",
    "I had an argument with my friend, but we'll work it out.",
    "I'm nervous about my upcoming presentation.",
    "I didn't sleep well last night.",
    "I'm feeling a bit under the weather today.",
    "I'm disappointed I didn't get that job.",
    "I'm worried about my exam next week.",
    "I miss my family who live far away.",
    "I feel lonely sometimes on weekends.",
    "I'm frustrated with my slow progress at the gym.",
    "I'm annoyed at myself for procrastinating again.",
    "I'm concerned about my finances this month.",
    "I feel a bit overwhelmed with my workload right now.",
    "I'm sad that my vacation is over.",
    "I'm having a tough time with this project.",
    "I feel insecure about my abilities sometimes.",
    "I'm upset that my plans got cancelled.",
    "I'm feeling a bit lost about my career direction.",
    "I'm having trouble making a difficult decision.",
    "I'm disappointed in myself for making that mistake.",
    "I feel like I'm not making enough progress in life.",
    "I'm stressed about my upcoming deadline.",
    "I feel like I need a break from everything.",
    "I'm worried about my health check-up next week.",
    "I'm feeling a bit down today for no particular reason.",
    "I'm anxious about meeting new people at the event.",
    "I'm having a hard time balancing work and personal life."
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
                variation = ' '.join(random.sample(words, len(words)))
                expanded.append(variation)
    return expanded

# Расширяем наборы текстов
suicidal_expanded = expand_dataset(suicidal_texts)
depressive_expanded = expand_dataset(depressive_texts)
normal_expanded = expand_dataset(normal_texts)

# Создаем DataFrame
data = []

# Добавляем суицидальные тексты (высокий риск)
for text in suicidal_expanded:
    data.append({
        'text': text,
        'label': 'suicidal',
        'risk_level': 'high'
    })

# Добавляем депрессивные тексты (средний риск)
for text in depressive_expanded:
    data.append({
        'text': text,
        'label': 'depression',
        'risk_level': 'medium'
    })

# Добавляем нормальные тексты (низкий риск)
for text in normal_expanded:
    data.append({
        'text': text,
        'label': 'normal',
        'risk_level': 'low'
    })

# Создаем DataFrame
df = pd.DataFrame(data)

# Перемешиваем данные
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Разделяем на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Сохраняем датасеты
train_df.to_csv('/home/ubuntu/mental_health_project/data/processed/train_data.csv', index=False)
test_df.to_csv('/home/ubuntu/mental_health_project/data/processed/test_data.csv', index=False)

# Выводим статистику
print(f"Всего примеров: {len(df)}")
print(f"Обучающая выборка: {len(train_df)}")
print(f"Тестовая выборка: {len(test_df)}")
print("\nРаспределение классов в обучающей выборке:")
print(train_df['label'].value_counts())
print("\nРаспределение уровней риска в обучающей выборке:")
print(train_df['risk_level'].value_counts())

print("\nДатасет успешно создан и сохранен в директории /home/ubuntu/mental_health_project/data/processed/")
