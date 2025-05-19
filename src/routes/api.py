"""
API маршруты для системы мониторинга психического здоровья
"""

from flask import Blueprint, request, jsonify
import os
import sys
from src.models.analyzer import MentalHealthAnalyzer

# Создаем Blueprint для API
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Инициализируем анализатор без загрузки модели из файла
# Используем встроенные эвристики для демонстрации
analyzer = MentalHealthAnalyzer()

@api_bp.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Анализ текстового сообщения
    """
    # Проверяем наличие JSON в запросе
    if not request.is_json:
        return jsonify({'error': 'Ожидается JSON'}), 400
    
    # Получаем данные из запроса
    data = request.get_json()
    
    # Проверяем наличие поля message
    if 'message' not in data or not data['message']:
        return jsonify({'error': 'Отсутствует текст для анализа'}), 400
    
    # Анализируем текст
    try:
        result = analyzer.analyze_text(data['message'])
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': f'Ошибка при анализе: {str(e)}'}), 500

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Проверка работоспособности API
    """
    return jsonify({'status': 'ok', 'message': 'API работает корректно'}), 200
