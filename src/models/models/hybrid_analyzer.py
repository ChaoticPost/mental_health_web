"""
Гибридный анализатор психического здоровья, объединяющий эвристику и ML-модель.
Реализует каскадную архитектуру для определения рисков психического здоровья.
"""

import os
import pickle
import numpy as np
from .heuristic_analyzer import HeuristicAnalyzer

class HybridAnalyzer:
    """
    Гибридный анализатор, объединяющий эвристику и ML-модель.
    """
    
    def __init__(self, model_path=None, heuristic_config_path=None):
        """
        Инициализация гибридного анализатора.
        
        Args:
            model_path (str, optional): Путь к файлу с обученной ML-моделью.
            heuristic_config_path (str, optional): Путь к файлу конфигурации эвристического анализатора.
        """
        # Инициализация эвристического анализатора
        self.heuristic_analyzer = HeuristicAnalyzer(heuristic_config_path)
        
        # Инициализация ML-анализатора
        self.ml_model = None
        self.vectorizer = None
        self.label_encoder = None
        self.label_to_risk = None
        
        # Загрузка ML-модели, если указан путь
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"Предупреждение: ML-модель не найдена по пути {model_path}")
    
    def _load_model(self, model_path):
        """
        Загрузка ML-модели из файла.
        
        Args:
            model_path (str): Путь к файлу с моделью.
        """
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.ml_model = model_data.get('model')
            self.vectorizer = model_data.get('vectorizer')
            self.label_encoder = model_data.get('label_encoder')
            self.label_to_risk = model_data.get('label_to_risk', {
                'suicidal': 'high',
                'depression': 'medium',
                'normal': 'low'
            })
            
            print(f"ML-модель успешно загружена из {model_path}")
        except Exception as e:
            print(f"Ошибка при загрузке ML-модели: {e}")
            self.ml_model = None
    
    def _analyze_with_ml(self, text):
        """
        Анализ текста с использованием ML-модели.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            dict: Результат анализа с уровнем риска и уверенностью.
        """
        if not self.ml_model or not self.vectorizer:
            return {
                'risk_level': 'unknown',
                'confidence': 0.0,
                'method': 'ml',
                'prediction': None,
                'probabilities': {}
            }
        
        try:
            # Предсказание
            X = self.vectorizer.transform([text])
            y_pred = self.ml_model.predict(X)[0]
            y_pred_proba = self.ml_model.predict_proba(X)[0]
            
            # Декодирование метки
            if hasattr(self.label_encoder, 'inverse_transform'):
                prediction = self.label_encoder.inverse_transform([y_pred])[0]
            else:
                prediction = str(y_pred)
            
            # Определение уровня риска
            risk_level = self.label_to_risk.get(prediction, 'unknown')
            
            # Определение уверенности
            confidence = max(y_pred_proba)
            
            # Формирование результата
            result = {
                'risk_level': risk_level,
                'confidence': confidence,
                'method': 'ml',
                'prediction': prediction,
                'probabilities': dict(zip(self.label_encoder.classes_, y_pred_proba))
            }
            
            return result
        except Exception as e:
            print(f"Ошибка при анализе с помощью ML-модели: {e}")
            return {
                'risk_level': 'unknown',
                'confidence': 0.0,
                'method': 'ml',
                'prediction': None,
                'probabilities': {}
            }
    
    def analyze_text(self, text):
        """
        Полный анализ текста с использованием гибридного подхода.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            dict: Результаты анализа.
        """
        if not text or not isinstance(text, str):
            return {
                'risk_level': 'unknown',
                'confidence': 0.0,
                'method': 'hybrid',
                'analysis': {
                    'heuristic': None,
                    'ml': None
                }
            }
        
        # Сначала применяем эвристический анализатор
        heuristic_result = self.heuristic_analyzer.analyze(text)
        
        # Если эвристика обнаружила высокий или средний риск с высокой уверенностью,
        # возвращаем результат без применения ML-модели
        if (heuristic_result['risk_level'] == 'high' or 
            (heuristic_result['risk_level'] == 'medium' and heuristic_result['confidence'] >= 0.75)):
            return {
                'risk_level': heuristic_result['risk_level'],
                'confidence': heuristic_result['confidence'],
                'method': 'heuristic',
                'trigger': heuristic_result.get('trigger'),
                'trigger_type': heuristic_result.get('trigger_type'),
                'analysis': {
                    'heuristic': heuristic_result,
                    'ml': None  # ML-модель не использовалась
                }
            }
        
        # Если эвристика не обнаружила высокий риск или уверенность низкая,
        # применяем ML-модель
        ml_result = self._analyze_with_ml(text)
        
        # Интеграция результатов
        final_result = self._integrate_results(heuristic_result, ml_result)
        
        return final_result
    
    def _integrate_results(self, heuristic_result, ml_result):
        """
        Интеграция результатов эвристики и ML-модели.
        
        Args:
            heuristic_result (dict): Результат эвристического анализа.
            ml_result (dict): Результат ML-анализа.
            
        Returns:
            dict: Интегрированный результат.
        """
        # Если ML-модель не использовалась или произошла ошибка
        if not ml_result or ml_result['risk_level'] == 'unknown':
            # Если эвристика дала результат, используем его
            if heuristic_result['risk_level'] != 'unknown':
                return {
                    'risk_level': heuristic_result['risk_level'],
                    'confidence': heuristic_result['confidence'],
                    'method': 'heuristic',
                    'trigger': heuristic_result.get('trigger'),
                    'trigger_type': heuristic_result.get('trigger_type'),
                    'analysis': {
                        'heuristic': heuristic_result,
                        'ml': ml_result
                    }
                }
            # Иначе возвращаем неизвестный риск
            else:
                return {
                    'risk_level': 'unknown',
                    'confidence': 0.0,
                    'method': 'hybrid',
                    'analysis': {
                        'heuristic': heuristic_result,
                        'ml': ml_result
                    }
                }
        
        # Приоритет отдается высокому риску
        if ml_result['risk_level'] == 'high':
            risk_level = 'high'
            confidence = ml_result['confidence']
            method = 'ml'
        # Если эвристика определила средний риск, а ML - низкий,
        # приоритет отдается эвристике при достаточной уверенности
        elif (heuristic_result['risk_level'] == 'medium' and 
              ml_result['risk_level'] == 'low' and 
              heuristic_result['confidence'] >= 0.6):
            risk_level = 'medium'
            confidence = heuristic_result['confidence']
            method = 'heuristic'
        # В остальных случаях используем результат ML-модели
        else:
            risk_level = ml_result['risk_level']
            confidence = ml_result['confidence']
            method = 'ml'
        
        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'method': method,
            'prediction': ml_result.get('prediction') if method == 'ml' else None,
            'trigger': heuristic_result.get('trigger') if method == 'heuristic' else None,
            'trigger_type': heuristic_result.get('trigger_type') if method == 'heuristic' else None,
            'analysis': {
                'heuristic': heuristic_result,
                'ml': ml_result
            }
        }
    
    def generate_recommendations(self, risk_level):
        """
        Генерация рекомендаций на основе уровня риска.
        
        Args:
            risk_level (str): Уровень риска ('low', 'medium', 'high').
            
        Returns:
            list: Список рекомендаций.
        """
        # Рекомендации по уровням риска
        recommendations = {
            'high': [
                "Рекомендуется немедленная консультация специалиста по психическому здоровью.",
                "Обратитесь на горячую линию психологической помощи.",
                "Не оставайтесь в одиночестве, свяжитесь с близким человеком.",
                "Если у вас есть суицидальные мысли, позвоните на линию предотвращения самоубийств."
            ],
            'medium': [
                "Рекомендуется консультация специалиста по психическому здоровью.",
                "Обратите внимание на режим сна и физическую активность.",
                "Практикуйте техники управления стрессом и релаксации.",
                "Поговорите с доверенным человеком о своих чувствах."
            ],
            'low': [
                "Продолжайте следить за своим психическим здоровьем.",
                "Практикуйте регулярные техники релаксации и осознанности.",
                "Поддерживайте здоровый образ жизни и социальные связи.",
                "Уделяйте время занятиям, которые приносят вам радость."
            ],
            'unknown': [
                "Недостаточно информации для определения уровня риска.",
                "Рекомендуется консультация специалиста для более точной оценки."
            ]
        }
        
        return recommendations.get(risk_level, recommendations['unknown'])

# Тестирование гибридного анализатора
if __name__ == "__main__":
    # Пути к файлам
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'ml', 'mental_health_model.pkl')
    config_path = os.path.join(current_dir, 'config', 'heuristic_config.json')
    
    # Инициализация анализатора
    analyzer = HybridAnalyzer(model_path, config_path)
    
    # Примеры текстов для тестирования
    test_texts = [
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
        "I feel so depressed lately, nothing brings me joy.",
        "I'm struggling to get out of bed every morning.",
        "Everything feels pointless now.",
        "I've lost interest in all the things I used to enjoy.",
        "I feel empty inside all the time.",
        "I'm so tired no matter how much I sleep.",
        "I had a bad day today, but tomorrow will be better.",
        "Work has been stressful, but I'm managing.",
        "I'm feeling a bit sad after watching that movie.",
        "I'm tired after a long day at work.",
        "I had an argument with my friend, but we'll work it out."
    ]
    
    # Тестирование
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"Текст: {text}")
        print(f"Результат: {result['risk_level']} (уверенность: {result['confidence']:.2f}, метод: {result['method']})")
        
        if result['method'] == 'heuristic' and result.get('trigger'):
            print(f"Триггер: {result['trigger']} (тип: {result.get('trigger_type', 'неизвестно')})")
        elif result['method'] == 'ml' and result.get('prediction'):
            print(f"Предсказание ML: {result['prediction']}")
        
        recommendations = analyzer.generate_recommendations(result['risk_level'])
        print("Рекомендации:")
        for rec in recommendations:
            print(f"- {rec}")
        
        print("-" * 80)
