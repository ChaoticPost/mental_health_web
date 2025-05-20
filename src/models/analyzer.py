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
    Предоставляет методы для оценки эмоционального состояния и уровня риска.
    """
    
    def __init__(self, model_path=None):
        """
        Инициализация анализатора психического здоровья.
        
        Args:
            model_path (str, optional): Путь к файлу с обученной моделью.
        """
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Расширенные списки слов для анализа
        self._initialize_word_lists()
        
        # Пороги чувствительности для анализа
        self.risk_thresholds = {
            'high': 0.75,  # Порог для высокого риска
            'medium': 0.5,  # Порог для среднего риска
            'low': 0.25     # Порог для низкого риска
        }
        
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
    
    def _initialize_word_lists(self):
        """
        Инициализация расширенных списков слов и фраз для анализа.
        """
        # Расширенный список негативных слов и фраз
        self.negative_words = [
            # Базовые негативные эмоции
            'sad', 'depressed', 'unhappy', 'angry', 'miserable', 'terrible', 'awful', 
            'horrible', 'bad', 'worse', 'worst', 'trouble', 'difficult', 'anxiety', 
            'anxious', 'worry', 'worried', 'stressed', 'stress', 'lonely', 'alone', 'isolated',
            
            # Дополнительные негативные слова
            'hopeless', 'helpless', 'worthless', 'useless', 'failure', 'pathetic',
            'disappointed', 'disappointing', 'frustrated', 'frustrating', 'irritated',
            'annoyed', 'upset', 'distressed', 'disturbed', 'devastated', 'heartbroken',
            'hurt', 'pain', 'painful', 'suffering', 'agony', 'agonizing', 'torment',
            'tortured', 'trapped', 'stuck', 'suffocating', 'drowning', 'dying',
            'exhausted', 'tired', 'fatigue', 'drained', 'empty', 'numb', 'hollow',
            'lost', 'confused', 'disoriented', 'overwhelmed', 'burdened', 'heavy',
            'dark', 'darkness', 'gloomy', 'bleak', 'grim', 'desperate', 'despairing',
            'suicidal', 'self-harm', 'self-hatred', 'self-loathing', 'guilt', 'shame',
            'regret', 'remorse', 'bitter', 'resentful', 'vengeful', 'hateful', 'hate',
            'disgusted', 'disgusting', 'repulsed', 'repulsive', 'sick', 'ill', 'unwell'
        ]
        
        # Расширенный список позитивных слов и фраз
        self.positive_words = [
            # Базовые позитивные эмоции
            'happy', 'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 
            'terrific', 'joy', 'joyful', 'content', 'satisfied', 'pleased', 'delighted', 
            'glad', 'cheerful', 'peaceful', 'calm', 'relaxed', 'hopeful',
            
            # Дополнительные позитивные слова
            'blessed', 'grateful', 'thankful', 'appreciative', 'inspired', 'motivated',
            'energetic', 'vibrant', 'alive', 'thriving', 'flourishing', 'growing',
            'improving', 'better', 'best', 'perfect', 'ideal', 'superb', 'outstanding',
            'exceptional', 'remarkable', 'impressive', 'admirable', 'respected', 'valued',
            'worthy', 'deserving', 'accomplished', 'successful', 'achieving', 'proud',
            'confident', 'strong', 'resilient', 'brave', 'courageous', 'fearless',
            'determined', 'persistent', 'dedicated', 'committed', 'focused', 'clear',
            'bright', 'brilliant', 'smart', 'intelligent', 'wise', 'insightful',
            'understanding', 'compassionate', 'empathetic', 'kind', 'loving', 'loved',
            'adored', 'cherished', 'treasured', 'special', 'unique', 'extraordinary',
            'incredible', 'unbelievable', 'miraculous', 'magical', 'enchanted', 'charmed'
        ]
        
        # Контекстуальные фразы для анализа
        self.contextual_phrases = {
            'negative': [
                'feel like giving up', 'want to give up', 'can\'t take it anymore',
                'no point in trying', 'no reason to live', 'better off dead',
                'want to die', 'thinking about suicide', 'considering suicide',
                'planning suicide', 'end my life', 'take my life', 'kill myself',
                'harm myself', 'hurt myself', 'cut myself', 'self harm',
                'don\'t want to be here', 'don\'t want to exist', 'wish I was dead',
                'wish I wasn\'t alive', 'wish I wasn\'t born', 'life is meaningless',
                'life is pointless', 'life is worthless', 'no hope', 'no future',
                'can\'t see a way out', 'trapped in darkness', 'drowning in sorrow',
                'consumed by sadness', 'overwhelmed by sadness', 'crushed by depression',
                'crippled by anxiety', 'paralyzed by fear', 'tormented by thoughts',
                'haunted by memories', 'plagued by nightmares', 'can\'t sleep',
                'can\'t eat', 'can\'t focus', 'can\'t concentrate', 'can\'t function',
                'falling apart', 'breaking down', 'losing control', 'losing my mind',
                'going crazy', 'going insane', 'nobody cares', 'nobody understands',
                'nobody loves me', 'everyone hates me', 'everyone would be better off without me',
                'I\'m a burden', 'I\'m worthless', 'I\'m useless', 'I\'m a failure',
                'I\'m not good enough', 'I\'m not worthy', 'I\'m not deserving',
                'I\'m not important', 'I\'m not needed', 'I\'m not wanted'
            ],
            'positive': [
                'feeling better', 'getting better', 'improving', 'making progress',
                'seeing improvement', 'seeing progress', 'feeling hopeful', 'have hope',
                'looking forward', 'excited about future', 'optimistic about future',
                'things are looking up', 'light at the end of the tunnel', 'silver lining',
                'grateful for', 'thankful for', 'appreciate', 'blessed with', 'lucky to have',
                'fortunate to have', 'happy with', 'content with', 'satisfied with',
                'proud of myself', 'proud of what I\'ve accomplished', 'achieved a lot',
                'made progress', 'overcome obstacles', 'conquered challenges',
                'survived difficult times', 'got through tough times', 'persevered through',
                'stayed strong', 'remained resilient', 'kept going', 'didn\'t give up',
                'found strength', 'found courage', 'found peace', 'found happiness',
                'found joy', 'found purpose', 'found meaning', 'life is good',
                'life is beautiful', 'life is worth living', 'enjoying life',
                'loving life', 'embracing life', 'living fully', 'living authentically',
                'being true to myself', 'being myself', 'accepting myself',
                'loving myself', 'taking care of myself', 'self-care', 'self-love',
                'self-compassion', 'self-acceptance', 'self-respect', 'self-worth'
            ]
        }
        
        # Усилители и смягчители для анализа
        self.intensifiers = [
            'very', 'extremely', 'incredibly', 'really', 'truly', 'absolutely',
            'completely', 'totally', 'utterly', 'deeply', 'profoundly', 'intensely',
            'severely', 'terribly', 'horribly', 'awfully', 'desperately', 'seriously',
            'exceptionally', 'extraordinarily', 'remarkably', 'particularly', 'especially',
            'notably', 'significantly', 'substantially', 'considerably', 'greatly',
            'highly', 'immensely', 'enormously', 'tremendously', 'exceedingly',
            'excessively', 'overly', 'unduly', 'unbelievably', 'inconceivably',
            'unimaginably', 'indescribably', 'unspeakably', 'unbearably', 'intolerably'
        ]
        
        self.diminishers = [
            'somewhat', 'slightly', 'a bit', 'a little', 'kind of', 'sort of',
            'rather', 'fairly', 'quite', 'relatively', 'moderately', 'reasonably',
            'partially', 'partly', 'somewhat', 'to some extent', 'to a degree',
            'in a way', 'in some ways', 'more or less', 'pretty much', 'almost',
            'nearly', 'practically', 'virtually', 'basically', 'essentially',
            'approximately', 'roughly', 'about', 'around', 'or so', 'give or take',
            'not very', 'not particularly', 'not especially', 'not notably',
            'not significantly', 'not substantially', 'not considerably',
            'not greatly', 'not terribly', 'not awfully', 'not exceptionally'
        ]
        
        self.negations = [
            'not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 'neither',
            'nor', 'hardly', 'scarcely', 'barely', 'rarely', 'seldom', 'don\'t',
            'doesn\'t', 'didn\'t', 'won\'t', 'wouldn\'t', 'can\'t', 'cannot',
            'couldn\'t', 'shouldn\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t',
            'haven\'t', 'hasn\'t', 'hadn\'t', 'without', 'lack', 'lacking', 'absent',
            'absence', 'free from', 'free of', 'void of', 'devoid of', 'clear of'
        ]
    
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
