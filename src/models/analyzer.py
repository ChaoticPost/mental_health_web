"""
Модуль для анализа психического здоровья по текстовым сообщениям.
Адаптирован для использования в веб-приложении Flask.
"""

import os
import re
import string
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Убедимся, что необходимые ресурсы NLTK загружены
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

class MentalHealthAnalyzer:
    """
    Класс для анализа психического здоровья по текстовым сообщениям.
    """
    
    def __init__(self, model_path=None):
        """
        Инициализация анализатора психического здоровья.
        
        Args:
            model_path (str, optional): Путь к файлу с обученной моделью.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Загрузка модели
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
                
            self.vectorizer = self.model_data.get('vectorizer')
            self.model = self.model_data.get('model')
            self.label_encoder = self.model_data.get('label_encoder')
        else:
            self.model_data = None
            self.vectorizer = None
            self.model = None
            self.label_encoder = None
            print(f"Предупреждение: Модель не найдена по пути {model_path}")
    
    def preprocess_text(self, text):
        """
        Предобработка текста для анализа.
        
        Args:
            text (str): Исходный текст.
            
        Returns:
            str: Предобработанный текст.
        """
        if not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Удаление HTML-тегов
        text = re.sub(r'<.*?>', '', text)
        
        # Удаление пунктуации
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Токенизация
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Если ресурсы не загружены, используем простую токенизацию по пробелам
            tokens = text.split()
        
        # Удаление стоп-слов и лемматизация
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text):
        """
        Анализ настроения текста (позитивное/негативное).
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            dict: Результат анализа настроения.
        """
        # Простая эвристика для демонстрации
        # В реальном приложении здесь должна быть интеграция с моделью анализа настроения
        
        negative_words = ['sad', 'depressed', 'unhappy', 'angry', 'miserable', 'terrible', 'awful', 
                         'horrible', 'bad', 'worse', 'worst', 'trouble', 'difficult', 'anxiety', 
                         'anxious', 'worry', 'stressed', 'stress', 'lonely', 'alone', 'isolated']
        
        positive_words = ['happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 
                         'terrific', 'joy', 'joyful', 'content', 'satisfied', 'pleased', 'delighted', 
                         'glad', 'cheerful', 'peaceful', 'calm', 'relaxed', 'hopeful']
        
        text_lower = text.lower()
        
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        total = neg_count + pos_count
        if total == 0:
            # Если нет явных индикаторов, используем длину текста как эвристику
            score = 0.5 + (len(text) % 10) / 20  # Случайное значение между 0.5 и 1.0
            label = "POSITIVE" if score > 0.7 else "NEGATIVE"
        else:
            if pos_count > neg_count:
                score = 0.5 + 0.5 * (pos_count / total)
                label = "POSITIVE"
            else:
                score = 0.5 + 0.5 * (neg_count / total)
                label = "NEGATIVE"
        
        return {
            "label": label,
            "score": min(0.99, max(0.6, score))  # Ограничиваем значение между 0.6 и 0.99
        }
    
    def analyze_emotions(self, text):
        """
        Анализ эмоций в тексте.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            dict: Словарь с оценками различных эмоций.
        """
        # Простая эвристика для демонстрации
        # В реальном приложении здесь должна быть интеграция с моделью анализа эмоций
        
        emotion_keywords = {
            'joy': ['happy', 'joy', 'delighted', 'glad', 'pleased', 'excited', 'cheerful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'heartbroken', 'gloomy'],
            'anger': ['angry', 'mad', 'furious', 'outraged', 'irritated', 'annoyed', 'frustrated'],
            'fear': ['afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'unexpected'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled', 'horrified']
        }
        
        text_lower = text.lower()
        results = {}
        
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for word in keywords if word in text_lower)
            # Нормализуем значение между 0 и 1 с некоторой случайностью
            base_score = min(1.0, count * 0.2)
            # Добавляем случайность для демонстрации
            random_factor = (hash(text + emotion) % 10) / 20  # Значение между 0 и 0.5
            score = min(0.95, max(0.05, base_score + random_factor))
            results[emotion] = score
        
        return results
    
    def predict(self, text):
        """
        Предсказание категории психического здоровья на основе текста.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            str: Предсказанная категория.
        """
        if not self.model or not self.vectorizer:
            # Если модель не загружена, возвращаем заглушку
            return "unknown"
        
        # Предобработка текста
        processed_text = self.preprocess_text(text)
        
        # Векторизация текста
        X = self.vectorizer.transform([processed_text])
        
        # Предсказание
        y_pred = self.model.predict(X)
        
        # Декодирование метки
        if hasattr(self.label_encoder, 'inverse_transform'):
            prediction = self.label_encoder.inverse_transform(y_pred)[0]
        else:
            prediction = str(y_pred[0])
        
        return prediction
    
    def assess_risk(self, text, prediction=None, sentiment=None):
        """
        Оценка уровня риска на основе текста и предсказания.
        
        Args:
            text (str): Текст для анализа.
            prediction (str, optional): Предсказанная категория.
            sentiment (dict, optional): Результат анализа настроения.
            
        Returns:
            str: Уровень риска ('low', 'medium', 'high').
        """
        # Если предсказание не предоставлено, получаем его
        if prediction is None:
            prediction = self.predict(text)
        
        # Если анализ настроения не предоставлен, получаем его
        if sentiment is None:
            sentiment = self.analyze_sentiment(text)
        
        # Ключевые слова высокого риска
        high_risk_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
            'no reason to live', 'can\'t go on', 'hopeless', 'worthless', 'burden',
            'never get better', 'unbearable', 'desperate', 'severe depression'
        ]
        
        # Ключевые слова среднего риска
        medium_risk_keywords = [
            'depressed', 'anxious', 'panic', 'overwhelmed', 'exhausted', 'stressed',
            'can\'t sleep', 'insomnia', 'no energy', 'tired all the time', 'worried',
            'fear', 'scared', 'lonely', 'isolated', 'sad', 'crying'
        ]
        
        # Проверка на ключевые слова высокого риска
        text_lower = text.lower()
        for keyword in high_risk_keywords:
            if keyword in text_lower:
                return 'high'
        
        # Проверка на ключевые слова среднего риска
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in text_lower)
        
        # Учитываем настроение
        is_negative = sentiment.get('label') == 'NEGATIVE'
        sentiment_score = sentiment.get('score', 0.5)
        
        # Определение уровня риска
        if medium_risk_count >= 3 and is_negative and sentiment_score > 0.8:
            return 'high'
        elif medium_risk_count >= 2 or (is_negative and sentiment_score > 0.7):
            return 'medium'
        else:
            return 'low'
    
    def generate_recommendations(self, risk_level):
        """
        Генерация рекомендаций на основе уровня риска.
        
        Args:
            risk_level (str): Уровень риска ('low', 'medium', 'high').
            
        Returns:
            list: Список рекомендаций.
        """
        if risk_level == 'high':
            return [
                "Рекомендуется немедленная консультация специалиста по психическому здоровью.",
                "Обратитесь на горячую линию психологической помощи.",
                "Не оставайтесь в одиночестве, свяжитесь с близким человеком."
            ]
        elif risk_level == 'medium':
            return [
                "Рекомендуется консультация специалиста по психическому здоровью.",
                "Обратите внимание на режим сна и физическую активность.",
                "Практикуйте техники управления стрессом и релаксации."
            ]
        else:  # low
            return [
                "Продолжайте следить за своим психическим здоровьем.",
                "Практикуйте регулярные техники релаксации и осознанности.",
                "Поддерживайте здоровый образ жизни и социальные связи."
            ]
    
    def analyze_text(self, text):
        """
        Полный анализ текста для оценки психического здоровья.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            dict: Результаты анализа.
        """
        # Предсказание категории
        prediction = self.predict(text)
        
        # Анализ настроения
        sentiment = self.analyze_sentiment(text)
        
        # Анализ эмоций
        emotions = self.analyze_emotions(text)
        
        # Оценка уровня риска
        risk_level = self.assess_risk(text, prediction, sentiment)
        
        # Генерация рекомендаций
        recommendations = self.generate_recommendations(risk_level)
        
        # Формирование результата
        result = {
            'risk_level': risk_level,
            'analysis': {
                'prediction': prediction,
                'sentiment': sentiment,
                'emotions': emotions
            },
            'recommendations': recommendations
        }
        
        return result
