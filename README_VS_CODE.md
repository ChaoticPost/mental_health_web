# Система мониторинга психического здоровья по текстовым сообщениям

## Описание проекта

Веб-приложение для анализа текстовых сообщений с целью выявления потенциальных проблем с психическим здоровьем. Система использует методы обработки естественного языка (NLP) и машинного обучения для анализа текста, определения настроения, эмоций и предоставления персонализированных рекомендаций.

## Структура проекта для VS Code

```
mental_health_web/
├── venv/                      # Виртуальное окружение Python (создается автоматически)
├── src/                       # Исходный код приложения
│   ├── models/                # Модели данных и ML-модели
│   │   ├── ml/                # Директория для ML-моделей
│   │   │   └── mental_health_model_real_data.pkl  # Обученная модель
│   │   └── analyzer.py        # Класс для анализа текста
│   ├── routes/                # Маршруты Flask
│   │   └── api.py             # API-маршруты для анализа текста
│   ├── static/                # Статические файлы
│   │   ├── css/               # CSS-стили
│   │   │   └── styles.css     # Основные стили приложения
│   │   ├── js/                # JavaScript-файлы
│   │   │   └── main.js        # Основной JS-файл для интерактивности
│   │   ├── index.html         # Главная страница
│   │   └── about.html         # Страница "О системе"
│   └── main.py                # Основной файл Flask-приложения
└── requirements.txt           # Зависимости проекта
```

## Настройка проекта в VS Code

### Шаг 1: Открытие проекта

1. Запустите VS Code
2. Выберите `File` > `Open Folder...` (или `Файл` > `Открыть папку...`)
3. Выберите директорию `mental_health_web`

### Шаг 2: Настройка виртуального окружения

1. Откройте терминал в VS Code (`Terminal` > `New Terminal` или `Терминал` > `Новый терминал`)
2. Создайте виртуальное окружение:
   ```bash
   python -m venv venv
   ```
3. Активируйте виртуальное окружение:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

### Шаг 3: Настройка интерпретатора Python

1. Нажмите `Ctrl+Shift+P` (или `Cmd+Shift+P` на macOS) для открытия палитры команд
2. Введите `Python: Select Interpreter` и выберите эту команду
3. Выберите интерпретатор из виртуального окружения (`venv`)

### Шаг 4: Установка рекомендуемых расширений

Для удобной работы с проектом рекомендуется установить следующие расширения VS Code:

1. Python (Microsoft) - для работы с Python
2. Flask-Snippets - для удобной работы с Flask
3. HTML CSS Support - для работы с HTML/CSS
4. JavaScript (ES6) code snippets - для работы с JavaScript
5. Live Server - для предпросмотра HTML-страниц

## Запуск проекта

### Локальный запуск

1. Откройте терминал в VS Code
2. Активируйте виртуальное окружение (если еще не активировано)
3. Запустите Flask-приложение:
   ```bash
   cd src
   python main.py
   ```
4. Откройте браузер и перейдите по адресу `http://localhost:5000`

### Запуск с помощью отладчика VS Code

1. Перейдите на вкладку "Run and Debug" (или "Запуск и отладка") в боковой панели VS Code
2. Нажмите на "create a launch.json file" (или "создать файл launch.json")
3. Выберите "Flask"
4. В созданном файле `.vscode/launch.json` убедитесь, что путь к `program` указывает на `src/main.py`
5. Нажмите F5 для запуска приложения в режиме отладки

## Структура кода и основные компоненты

### 1. Основной модуль Flask (`src/main.py`)

Этот файл является точкой входа в приложение. Он создает экземпляр Flask, регистрирует маршруты и запускает сервер.

```python
from flask import Flask, render_template, send_from_directory
from src.routes.api import api_bp

app = Flask(__name__)
app.register_blueprint(api_bp)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 2. API-маршруты (`src/routes/api.py`)

Этот файл содержит API-маршруты для анализа текста. Он создает Blueprint для API и определяет маршрут `/api/analyze` для обработки POST-запросов.

```python
from flask import Blueprint, request, jsonify
from src.models.analyzer import MentalHealthAnalyzer

api_bp = Blueprint('api', __name__, url_prefix='/api')
analyzer = MentalHealthAnalyzer()

@api_bp.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    result = analyzer.analyze_text(data['message'])
    return jsonify(result), 200
```

### 3. Анализатор текста (`src/models/analyzer.py`)

Этот файл содержит класс `MentalHealthAnalyzer`, который выполняет анализ текста, определение настроения, эмоций и генерацию рекомендаций.

```python
class MentalHealthAnalyzer:
    def __init__(self, model_path=None):
        # Инициализация анализатора
        
    def analyze_text(self, text):
        # Анализ текста и формирование результата
        
    def preprocess_text(self, text):
        # Предобработка текста
        
    def analyze_sentiment(self, text):
        # Анализ настроения
        
    def analyze_emotions(self, text):
        # Анализ эмоций
```

### 4. Фронтенд

#### HTML (`src/static/index.html`, `src/static/about.html`)

HTML-файлы определяют структуру веб-страниц. `index.html` содержит форму для ввода текста и область для отображения результатов анализа. `about.html` содержит информацию о системе.

#### CSS (`src/static/css/styles.css`)

CSS-файл определяет стили для веб-страниц, включая цвета, шрифты, отступы и анимации.

#### JavaScript (`src/static/js/main.js`)

JavaScript-файл обеспечивает интерактивность веб-страниц. Он отправляет AJAX-запросы к API, обрабатывает ответы и обновляет DOM.

## Рекомендации по разработке

### Добавление новых функций

1. **Добавление новых API-маршрутов**:
   - Создайте новый маршрут в `src/routes/api.py`
   - Добавьте соответствующую функцию в `src/models/analyzer.py`

2. **Улучшение анализа текста**:
   - Модифицируйте методы в классе `MentalHealthAnalyzer`
   - Добавьте новые методы для дополнительного анализа

3. **Улучшение фронтенда**:
   - Модифицируйте HTML-файлы для изменения структуры страниц
   - Обновите CSS-файл для изменения стилей
   - Модифицируйте JavaScript-файл для изменения интерактивности

### Интеграция с реальной ML-моделью

Текущая версия использует эвристические алгоритмы для демонстрации. Для интеграции с реальной ML-моделью:

1. Поместите обученную модель в директорию `src/models/ml/`
2. Модифицируйте метод `__init__` в классе `MentalHealthAnalyzer` для загрузки модели
3. Обновите методы анализа для использования загруженной модели

## Отладка и тестирование

### Отладка в VS Code

1. Установите точки останова (breakpoints) в коде, нажав на левый край редактора рядом с номером строки
2. Запустите приложение в режиме отладки (F5)
3. Используйте панель отладки для просмотра переменных, стека вызовов и т.д.

### Тестирование API

Для тестирования API можно использовать расширение VS Code "REST Client" или внешние инструменты, такие как Postman:

```http
POST http://localhost:5000/api/analyze
Content-Type: application/json

{
    "message": "I've been feeling really down lately and can't seem to focus on work."
}
```

## Деплой приложения

Для деплоя приложения на продакшн-сервер:

1. Убедитесь, что `debug=False` в `src/main.py`
2. Используйте WSGI-сервер, такой как Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 src.main:app
   ```
3. Настройте Nginx или Apache в качестве прокси-сервера

## Заключение

Этот проект предоставляет основу для системы мониторинга психического здоровья по текстовым сообщениям. Он может быть расширен и улучшен для использования в реальных сценариях.

## Контакты

Проект создан в рамках ВКР.  
По вопросам пишите в [Telegram → @daria_chugu](https://t.me/daria_chugu)