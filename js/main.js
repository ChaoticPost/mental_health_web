// Main JavaScript for Mental Health Monitoring System

document.addEventListener('DOMContentLoaded', function() {
    const analysisForm = document.getElementById('analysis-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultsCard = document.getElementById('results-card');
    const riskAlert = document.getElementById('risk-alert');
    const riskLevel = document.getElementById('risk-level');
    const riskDescription = document.getElementById('risk-description');
    const prediction = document.getElementById('prediction');
    const sentiment = document.getElementById('sentiment');
    const recommendations = document.getElementById('recommendations');
    
    let emotionsChart = null;

    analysisForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = document.getElementById('message').value.trim();
        
        if (!message) {
            alert('Пожалуйста, введите текст для анализа');
            return;
        }
        
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Анализ...';
        
        // Hide previous results
        resultsCard.classList.add('d-none');
        
        // Send request to API
        fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Ошибка сервера');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Произошла ошибка при анализе сообщения. Пожалуйста, попробуйте еще раз.');
        })
        .finally(() => {
            // Reset button state
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="bi bi-search me-2"></i>Анализировать';
        });
    });
    
    function displayResults(data) {
        // Set risk level
        riskAlert.className = 'alert';
        riskAlert.classList.add(`alert-${data.risk_level === 'high' ? 'high' : data.risk_level === 'medium' ? 'medium' : 'low'}`);
        riskAlert.classList.remove('d-none');
        
        riskLevel.textContent = data.risk_level === 'high' ? 'Высокий' : 
                               data.risk_level === 'medium' ? 'Средний' : 'Низкий';
        
        // Set risk description
        if (data.risk_level === 'high') {
            riskDescription.textContent = 'Обнаружены признаки серьезных проблем с психическим здоровьем. Рекомендуется обратиться к специалисту.';
        } else if (data.risk_level === 'medium') {
            riskDescription.textContent = 'Обнаружены некоторые признаки проблем с психическим здоровьем. Рекомендуется обратить внимание на свое состояние.';
        } else {
            riskDescription.textContent = 'Серьезных проблем с психическим здоровьем не обнаружено.';
        }
        
        // Set prediction
        prediction.textContent = data.analysis.prediction || 'Нет данных';
        
        // Set sentiment
        if (data.analysis.sentiment) {
            const sentimentText = data.analysis.sentiment.label === 'POSITIVE' ? 'Позитивное' : 'Негативное';
            const confidence = (data.analysis.sentiment.score * 100).toFixed(1);
            sentiment.textContent = `${sentimentText} (уверенность: ${confidence}%)`;
        } else {
            sentiment.textContent = 'Нет данных';
        }
        
        // Set recommendations
        recommendations.innerHTML = '';
        if (data.recommendations && data.recommendations.length > 0) {
            data.recommendations.forEach(rec => {
                const li = document.createElement('li');
                li.className = 'list-group-item';
                li.textContent = rec;
                recommendations.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.className = 'list-group-item';
            li.textContent = 'Нет рекомендаций';
            recommendations.appendChild(li);
        }
        
        // Create emotions chart if emotions data exists
        if (data.analysis.emotions) {
            createEmotionsChart(data.analysis.emotions);
        }
        
        // Show results
        resultsCard.classList.remove('d-none');
        resultsCard.classList.add('show');
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    function createEmotionsChart(emotions) {
        // Destroy previous chart if exists
        if (emotionsChart) {
            emotionsChart.destroy();
        }
        
        const ctx = document.getElementById('emotions-chart').getContext('2d');
        
        // Convert emotions object to arrays for chart
        const labels = Object.keys(emotions);
        const values = Object.values(emotions);
        
        // Create chart
        emotionsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Уровень эмоции',
                    data: values,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    }
});
