"""
Главный модуль Flask-приложения для системы мониторинга психического здоровья
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # DON'T CHANGE THIS !!!

from flask import Flask, render_template, send_from_directory
from src.routes.api import api_bp

# Создаем экземпляр Flask
app = Flask(__name__)

# Регистрируем Blueprint для API
app.register_blueprint(api_bp)

# Маршрут для главной страницы
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Маршрут для страницы "О системе"
@app.route('/about')
def about():
    return send_from_directory('static', 'about.html')

# Обработчик ошибки 404
@app.errorhandler(404)
def page_not_found(e):
    return send_from_directory('static', '404.html'), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
