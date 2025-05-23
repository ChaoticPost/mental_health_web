"""
Улучшенный эвристический анализатор для определения рисков психического здоровья.
Используется в гибридной архитектуре для надежного выявления критических паттернов.
"""

import re
import os
import json

class HeuristicAnalyzer:
    """
    Анализатор на основе правил и шаблонов для выявления критических случаев.
    """
    
    def __init__(self, config_path=None):
        """
        Инициализация эвристического анализатора.
        
        Args:
            config_path (str, optional): Путь к файлу конфигурации с ключевыми словами и паттернами.
        """
        # Загрузка конфигурации, если указан путь
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.high_risk_phrases = config.get('high_risk_phrases', [])
                self.high_risk_patterns = config.get('high_risk_patterns', [])
                self.medium_risk_phrases = config.get('medium_risk_phrases', [])
                self.medium_risk_patterns = config.get('medium_risk_patterns', [])
        else:
            # Инициализация с дефолтными значениями
            self._initialize_default_values()
    
    def _initialize_default_values(self):
        """
        Инициализация дефолтных значений для ключевых слов и паттернов.
        """
        # Критические ключевые слова и фразы высокого риска
        self.high_risk_phrases = [
            'want to die', 'kill myself', 'end my life', 'suicide', 'suicidal',
            'no reason to live', 'better off dead', 'plan to kill', 'want to end it all',
            'going to kill myself', 'take my own life', 'don\'t want to live anymore',
            'want to disappear forever', 'ready to die', 'life is not worth living',
            'can\'t go on anymore', 'no point in living', 'no longer see the point of living',
            'nobody would miss me', 'everyone would be better off without me',
            'i\'m a burden to everyone', 'i\'ve been giving away my things',
            'writing a suicide note', 'saying goodbye', 'final goodbye',
            'tried everything nothing helps', 'no hope left', 'can\'t take it anymore',
            'want the pain to stop forever', 'there\'s no way out'
        ]
        
        # Регулярные выражения для сложных паттернов высокого риска
        self.high_risk_patterns = [
            r'(?i)plan(ning)?\s+(to|for)\s+(die|suicide|kill\s+myself)',
            r'(?i)(wrote|writing|written)\s+(suicide|goodbye)\s+(note|letter)',
            r'(?i)(gave|giving|given)\s+away\s+(my|all)\s+(things|possessions|belongings)',
            r'(?i)(no|don\'t\s+see(\s+a)?)\s+(point|reason|purpose)\s+(in|of|to|for)\s+(living|life|existing|going\s+on)',
            r'(?i)(can\'t|cannot|couldn\'t)\s+(take|handle|bear|stand)\s+(it|this|the\s+pain|the\s+suffering)\s+(anymore|any\s+longer)',
            r'(?i)(better|best)\s+(if\s+i|that\s+i|for\s+everyone\s+if\s+i)\s+(was|were|am)\s+(dead|gone|not\s+here|not\s+alive)',
            r'(?i)(nobody|no\s+one)\s+would\s+(care|notice|mind|miss\s+me)\s+(if|when)\s+i\s+(die|was\s+gone|wasn\'t\s+here)',
            r'(?i)(just|only)\s+want\s+(the\s+pain|this|it|everything)\s+to\s+(stop|end|be\s+over)',
            r'(?i)(thinking|thought|been\s+thinking)\s+(about|of)\s+(killing\s+myself|suicide|ending\s+it\s+all)',
            r'(?i)(collected|collecting|stockpiling|saved|saving)\s+(pills|medication|drugs|rope)',
            r'(?i)(researched|researching|looked\s+up|searching\s+for)\s+(ways|methods|how)\s+to\s+(kill\s+myself|commit\s+suicide|die)',
            r'(?i)(burden|weight|problem|trouble)\s+(to|for|on)\s+(everyone|my\s+family|my\s+friends|others)',
            r'(?i)(set|picked|chosen)\s+(a\s+date|the\s+time|when)\s+(to|for)\s+(die|end\s+it|kill\s+myself)',
            r'(?i)(tried|attempting|tried)\s+(therapy|medication|treatment|everything)\s+(nothing|doesn\'t|won\'t)\s+(works|help)',
            r'(?i)(life|living)\s+(is|feels|seems)\s+(pointless|meaningless|hopeless|worthless)',
            r'(?i)(don\'t|do\s+not)\s+(want\s+to|wish\s+to)\s+(wake\s+up|be\s+alive|exist|live)\s+(tomorrow|anymore|any\s+longer)',
            r'(?i)(wish|want|hope)\s+(i|to)\s+(was|were|wasn\'t|would\s+be)\s+(never\s+born|dead|gone)',
            r'(?i)(cutting|harming|hurting)\s+(myself|my\s+body|my\s+skin|my\s+arms|my\s+wrists)'
        ]
        
        # Ключевые слова и фразы среднего риска
        self.medium_risk_phrases = [
            'depressed', 'depression', 'anxious', 'anxiety', 'hopeless', 'helpless',
            'worthless', 'useless', 'failure', 'can\'t sleep', 'insomnia', 'no energy',
            'tired all the time', 'exhausted', 'overwhelmed', 'stressed', 'worried',
            'fear', 'scared', 'lonely', 'isolated', 'sad', 'crying', 'unhappy',
            'lost interest', 'no joy', 'empty inside', 'numb', 'can\'t focus',
            'can\'t concentrate', 'no motivation', 'giving up', 'struggling',
            'dark thoughts', 'negative thoughts', 'hate myself', 'self-loathing',
            'self-hatred', 'disappointed', 'frustrated', 'angry', 'irritable',
            'mood swings', 'emotional', 'can\'t cope', 'can\'t handle'
        ]
        
        # Регулярные выражения для сложных паттернов среднего риска
        self.medium_risk_patterns = [
            r'(?i)(feel|feeling|felt)\s+(so|very|really|extremely)\s+(sad|down|depressed|unhappy)',
            r'(?i)(lost|losing)\s+(interest|motivation|desire)\s+(in|for)\s+(things|activities|hobbies)',
            r'(?i)(don\'t|do\s+not)\s+(enjoy|like|care\s+about)\s+(anything|things)\s+(anymore|any\s+longer)',
            r'(?i)(can\'t|cannot|couldn\'t|unable\s+to)\s+(sleep|eat|focus|concentrate|function)',
            r'(?i)(always|constantly)\s+(tired|exhausted|fatigued|drained)',
            r'(?i)(feel|feeling|felt)\s+(overwhelmed|stressed|anxious|worried)',
            r'(?i)(crying|cry|cried)\s+(all\s+the\s+time|constantly|everyday|a\s+lot)',
            r'(?i)(feel|feeling|felt)\s+(empty|hollow|numb|nothing)',
            r'(?i)(hate|hating|hated)\s+(myself|my\s+life|my\s+body|who\s+i\s+am)',
            r'(?i)(no|zero|little)\s+(energy|motivation|drive|desire)\s+(to|for)\s+(do|doing)\s+(anything|things)',
            r'(?i)(isolating|isolated|avoiding|withdrawing)\s+(myself|from\s+others|from\s+friends|from\s+family)',
            r'(?i)(struggle|struggling|hard|difficult)\s+(to|with)\s+(get\s+through|make\s+it\s+through)\s+(the\s+day|each\s+day)',
            r'(?i)(feel|feeling|felt)\s+like\s+(a\s+failure|worthless|useless|a\s+burden)',
            r'(?i)(don\'t|do\s+not)\s+(see|have)\s+(hope|a\s+future|anything\s+to\s+look\s+forward\s+to)'
        ]
    
    def analyze(self, text):
        """
        Анализ текста на наличие критических паттернов.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            dict: Результат анализа с уровнем риска и уверенностью.
        """
        if not text or not isinstance(text, str):
            return {
                'risk_level': 'unknown',
                'confidence': 0.0,
                'method': 'heuristic',
                'trigger': None
            }
        
        text_lower = text.lower()
        
        # Проверка на критические фразы высокого риска
        for phrase in self.high_risk_phrases:
            if phrase in text_lower:
                return {
                    'risk_level': 'high',
                    'confidence': 0.95,
                    'method': 'heuristic',
                    'trigger': phrase,
                    'trigger_type': 'phrase'
                }
        
        # Проверка на критические паттерны высокого риска
        for pattern in self.high_risk_patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    'risk_level': 'high',
                    'confidence': 0.9,
                    'method': 'heuristic',
                    'trigger': match.group(0),
                    'trigger_type': 'pattern'
                }
        
        # Проверка на фразы среднего риска
        medium_risk_phrase_count = 0
        matched_medium_phrases = []
        
        for phrase in self.medium_risk_phrases:
            if phrase in text_lower:
                medium_risk_phrase_count += 1
                matched_medium_phrases.append(phrase)
                
                # Если найдено несколько фраз среднего риска, классифицируем как средний риск
                if medium_risk_phrase_count >= 2:
                    return {
                        'risk_level': 'medium',
                        'confidence': 0.8,
                        'method': 'heuristic',
                        'trigger': matched_medium_phrases,
                        'trigger_type': 'multiple_phrases'
                    }
        
        # Проверка на паттерны среднего риска
        for pattern in self.medium_risk_patterns:
            match = re.search(pattern, text)
            if match:
                return {
                    'risk_level': 'medium',
                    'confidence': 0.75,
                    'method': 'heuristic',
                    'trigger': match.group(0),
                    'trigger_type': 'pattern'
                }
        
        # Если найдена одна фраза среднего риска, классифицируем как средний риск с низкой уверенностью
        if medium_risk_phrase_count == 1:
            return {
                'risk_level': 'medium',
                'confidence': 0.6,
                'method': 'heuristic',
                'trigger': matched_medium_phrases[0],
                'trigger_type': 'phrase'
            }
        
        # Если критические паттерны не обнаружены
        return {
            'risk_level': 'unknown',
            'confidence': 0.0,
            'method': 'heuristic',
            'trigger': None,
            'trigger_type': None
        }
    
    def save_config(self, config_path):
        """
        Сохранение конфигурации в файл.
        
        Args:
            config_path (str): Путь для сохранения конфигурации.
        """
        config = {
            'high_risk_phrases': self.high_risk_phrases,
            'high_risk_patterns': self.high_risk_patterns,
            'medium_risk_phrases': self.medium_risk_phrases,
            'medium_risk_patterns': self.medium_risk_patterns
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Конфигурация сохранена в {config_path}")

# Тестирование эвристического анализатора
if __name__ == "__main__":
    analyzer = HeuristicAnalyzer()
    
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
        result = analyzer.analyze(text)
        print(f"Текст: {text}")
        print(f"Результат: {result['risk_level']} (уверенность: {result['confidence']})")
        if result['trigger']:
            print(f"Триггер: {result['trigger']} (тип: {result['trigger_type']})")
        print("-" * 80)
    
    # Сохранение конфигурации
    config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')
    os.makedirs(config_dir, exist_ok=True)
    analyzer.save_config(os.path.join(config_dir, 'heuristic_config.json'))
