"""
API маршруты для системы мониторинга психического здоровья.
Предоставляет эндпоинты для анализа текстовых сообщений.
"""

from flask import Blueprint, request, jsonify
from src.models.analyzer import MentalHealthAnalyzer

# Создаем Blueprint для API
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Инициализируем анализатор
# В продакшн-версии здесь можно указать путь к модели
analyzer = MentalHealthAnalyzer()

@api_bp.route('/analyze', methods=['POST'])
def analyze_text():
    """
    Анализ текстового сообщения.
    
    Ожидает JSON с полем 'message', содержащим текст для анализа.
    Возвращает результаты анализа психического здоровья.
    
    Returns:
        JSON: Результаты анализа или сообщение об ошибке.
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
    Проверка работоспособности API.
    
    Используется для мониторинга доступности сервиса.
    
    Returns:
        JSON: Статус работоспособности API.
    """
    return jsonify({'status': 'ok', 'message': 'API работает корректно'}), 200
