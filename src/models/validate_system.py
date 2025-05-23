"""
Скрипт для валидации гибридной системы мониторинга психического здоровья
на тестовых фразах.
"""

import os
import sys
import json
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import classification_report, confusion_matrix

# Добавляем путь к модулям проекта
sys.path.append('/home/ubuntu/mental_health_project')

# Импортируем гибридный анализатор
from src.models.hybrid_analyzer import HybridAnalyzer

# Пути к файлам
MODEL_DIR = '/home/ubuntu/mental_health_project/src/models/ml'
CONFIG_DIR = '/home/ubuntu/mental_health_project/src/models/config'
RESULTS_DIR = '/home/ubuntu/mental_health_project/results'

# Создаем директории, если они не существуют
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Тестовые фразы с ожидаемыми уровнями риска
test_phrases = [
    # Высокий риск - явные суицидальные высказывания
    {"text": "I want to kill myself, I can't take it anymore.", "expected_risk": "high"},
    {"text": "I'm planning to end my life tonight, I've written my note.", "expected_risk": "high"},
    {"text": "I don't see any reason to continue living, everything is pointless.", "expected_risk": "high"},
    {"text": "I no longer see the point of living.", "expected_risk": "high"},  # Ключевой пример из задания
    {"text": "I've been thinking about suicide a lot lately.", "expected_risk": "high"},
    
    # Высокий риск - завуалированные суицидальные высказывания
    {"text": "I just want this pain to stop forever.", "expected_risk": "high"},
    {"text": "Nobody would miss me if I was gone.", "expected_risk": "high"},
    {"text": "I've been giving away my things lately, won't need them soon.", "expected_risk": "high"},
    {"text": "I just want to sleep and never wake up.", "expected_risk": "high"},
    {"text": "I've tried everything, nothing helps, I want to die.", "expected_risk": "high"},
    
    # Средний риск - депрессивные высказывания
    {"text": "I feel so depressed lately, nothing brings me joy.", "expected_risk": "medium"},
    {"text": "I'm struggling to get out of bed every morning.", "expected_risk": "medium"},
    {"text": "Everything feels pointless now.", "expected_risk": "medium"},
    {"text": "I've lost interest in all the things I used to enjoy.", "expected_risk": "medium"},
    {"text": "I feel empty inside all the time.", "expected_risk": "medium"},
    
    # Низкий риск - нормальные высказывания
    {"text": "I had a bad day today, but tomorrow will be better.", "expected_risk": "low"},
    {"text": "Work has been stressful, but I'm managing.", "expected_risk": "low"},
    {"text": "I'm feeling a bit sad after watching that movie.", "expected_risk": "low"},
    {"text": "I'm tired after a long day at work.", "expected_risk": "low"},
    {"text": "I had an argument with my friend, but we'll work it out.", "expected_risk": "low"}
]

# Функция для валидации гибридной системы
def validate_hybrid_system():
    # Инициализация анализатора
    model_path = os.path.join(MODEL_DIR, 'mental_health_model.pkl')
    config_path = os.path.join(CONFIG_DIR, 'heuristic_config.json')
    
    # Проверяем наличие файла конфигурации
    if not os.path.exists(config_path):
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Инициализируем анализатор без конфигурации
        analyzer = HybridAnalyzer(model_path)
        
        # Сохраняем конфигурацию
        analyzer.heuristic_analyzer.save_config(config_path)
    else:
        analyzer = HybridAnalyzer(model_path, config_path)
    
    # Результаты анализа
    results = []
    
    # Анализ тестовых фраз
    for phrase in test_phrases:
        text = phrase["text"]
        expected_risk = phrase["expected_risk"]
        
        # Анализ текста
        result = analyzer.analyze_text(text)
        
        # Добавление результата
        results.append({
            "text": text,
            "expected_risk": expected_risk,
            "predicted_risk": result["risk_level"],
            "confidence": result["confidence"],
            "method": result["method"],
            "correct": result["risk_level"] == expected_risk
        })
    
    return results

# Функция для вычисления метрик
def calculate_metrics(results):
    # Преобразуем результаты в формат для sklearn
    y_true = [r["expected_risk"] for r in results]
    y_pred = [r["predicted_risk"] for r in results]
    
    # Вычисляем метрики
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Вычисляем общую точность
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    
    # Вычисляем точность по методам
    heuristic_results = [r for r in results if r["method"] == "heuristic"]
    ml_results = [r for r in results if r["method"] == "ml"]
    
    heuristic_accuracy = sum(1 for r in heuristic_results if r["correct"]) / len(heuristic_results) if heuristic_results else 0
    ml_accuracy = sum(1 for r in ml_results if r["correct"]) / len(ml_results) if ml_results else 0
    
    # Вычисляем точность по уровням риска
    high_risk_results = [r for r in results if r["expected_risk"] == "high"]
    medium_risk_results = [r for r in results if r["expected_risk"] == "medium"]
    low_risk_results = [r for r in results if r["expected_risk"] == "low"]
    
    high_risk_accuracy = sum(1 for r in high_risk_results if r["correct"]) / len(high_risk_results) if high_risk_results else 0
    medium_risk_accuracy = sum(1 for r in medium_risk_results if r["correct"]) / len(medium_risk_results) if medium_risk_results else 0
    low_risk_accuracy = sum(1 for r in low_risk_results if r["correct"]) / len(low_risk_results) if low_risk_results else 0
    
    # Формируем метрики
    metrics = {
        "overall_accuracy": accuracy,
        "heuristic_accuracy": heuristic_accuracy,
        "ml_accuracy": ml_accuracy,
        "high_risk_accuracy": high_risk_accuracy,
        "medium_risk_accuracy": medium_risk_accuracy,
        "low_risk_accuracy": low_risk_accuracy,
        "classification_report": report
    }
    
    return metrics

# Функция для сохранения результатов
def save_results(results, metrics):
    # Создаем DataFrame с результатами
    df = pd.DataFrame(results)
    
    # Сохраняем результаты в CSV
    df.to_csv(os.path.join(RESULTS_DIR, 'validation_results.csv'), index=False)
    
    # Сохраняем метрики в JSON
    with open(os.path.join(RESULTS_DIR, 'validation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Создаем отчет в Markdown
    with open(os.path.join(RESULTS_DIR, 'validation_report.md'), 'w') as f:
        f.write("# Отчет о валидации гибридной системы мониторинга психического здоровья\n\n")
        
        f.write("## Общие метрики\n\n")
        f.write(f"- Общая точность: {metrics['overall_accuracy']:.2%}\n")
        f.write(f"- Точность эвристики: {metrics['heuristic_accuracy']:.2%}\n")
        f.write(f"- Точность ML-модели: {metrics['ml_accuracy']:.2%}\n\n")
        
        f.write("## Точность по уровням риска\n\n")
        f.write(f"- Высокий риск: {metrics['high_risk_accuracy']:.2%}\n")
        f.write(f"- Средний риск: {metrics['medium_risk_accuracy']:.2%}\n")
        f.write(f"- Низкий риск: {metrics['low_risk_accuracy']:.2%}\n\n")
        
        f.write("## Классификационный отчет\n\n")
        report = metrics['classification_report']
        f.write("| Класс | Precision | Recall | F1-score | Support |\n")
        f.write("|-------|-----------|--------|----------|--------|\n")
        for cls in ['high', 'medium', 'low']:
            if cls in report:
                cls_report = report[cls]
                f.write(f"| {cls} | {cls_report['precision']:.2f} | {cls_report['recall']:.2f} | {cls_report['f1-score']:.2f} | {cls_report['support']} |\n")
        f.write("\n")
        
        f.write("## Результаты по фразам\n\n")
        f.write("| Текст | Ожидаемый риск | Предсказанный риск | Уверенность | Метод | Корректно |\n")
        f.write("|-------|---------------|-------------------|------------|-------|----------|\n")
        for r in results:
            f.write(f"| {r['text'][:50]}... | {r['expected_risk']} | {r['predicted_risk']} | {r['confidence']:.2f} | {r['method']} | {'✓' if r['correct'] else '✗'} |\n")
    
    print(f"Результаты сохранены в директории {RESULTS_DIR}")
    
    return os.path.join(RESULTS_DIR, 'validation_report.md')

# Функция для вывода результатов в консоль
def print_results(results, metrics):
    # Выводим общие метрики
    print("\n=== Общие метрики ===")
    print(f"Общая точность: {metrics['overall_accuracy']:.2%}")
    print(f"Точность эвристики: {metrics['heuristic_accuracy']:.2%}")
    print(f"Точность ML-модели: {metrics['ml_accuracy']:.2%}")
    
    # Выводим точность по уровням риска
    print("\n=== Точность по уровням риска ===")
    print(f"Высокий риск: {metrics['high_risk_accuracy']:.2%}")
    print(f"Средний риск: {metrics['medium_risk_accuracy']:.2%}")
    print(f"Низкий риск: {metrics['low_risk_accuracy']:.2%}")
    
    # Выводим классификационный отчет
    print("\n=== Классификационный отчет ===")
    report = metrics['classification_report']
    headers = ["Класс", "Precision", "Recall", "F1-score", "Support"]
    table = []
    for cls in ['high', 'medium', 'low']:
        if cls in report:
            cls_report = report[cls]
            table.append([
                cls,
                f"{cls_report['precision']:.2f}",
                f"{cls_report['recall']:.2f}",
                f"{cls_report['f1-score']:.2f}",
                cls_report['support']
            ])
    print(tabulate(table, headers=headers, tablefmt="grid"))
    
    # Выводим результаты по фразам
    print("\n=== Результаты по фразам ===")
    headers = ["Текст", "Ожидаемый риск", "Предсказанный риск", "Уверенность", "Метод", "Корректно"]
    table = []
    for r in results:
        table.append([
            r['text'][:50] + "...",
            r['expected_risk'],
            r['predicted_risk'],
            f"{r['confidence']:.2f}",
            r['method'],
            "✓" if r['correct'] else "✗"
        ])
    print(tabulate(table, headers=headers, tablefmt="grid"))

# Основная функция
def main():
    print("Начинаем валидацию гибридной системы...")
    
    # Валидация системы
    results = validate_hybrid_system()
    
    # Вычисление метрик
    metrics = calculate_metrics(results)
    
    # Вывод результатов
    print_results(results, metrics)
    
    # Сохранение результатов
    report_path = save_results(results, metrics)
    
    print(f"\nВалидация завершена. Отчет сохранен в {report_path}")
    
    return report_path

if __name__ == "__main__":
    main()
