from typing import Dict, List
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('course_recommendations.log'),
        logging.StreamHandler()
    ]
)

# База данных курсов
COURSES = {
    "python": {
        "beginner": "https://www.coursera.org/learn/python",
        "intermediate": "https://www.coursera.org/learn/intermediate-python",
        "advanced": "https://www.coursera.org/learn/advanced-python"
    },
    "machine learning": {
        "beginner": "https://www.coursera.org/learn/machine-learning",
        "intermediate": "https://www.coursera.org/learn/advanced-machine-learning",
        "advanced": "https://www.coursera.org/learn/deep-learning"
    },
    "deep learning": {
        "beginner": "https://www.coursera.org/learn/neural-networks-deep-learning",
        "intermediate": "https://www.coursera.org/learn/convolutional-neural-networks",
        "advanced": "https://www.coursera.org/learn/sequence-models"
    },
    "data science": {
        "beginner": "https://www.coursera.org/learn/data-science",
        "intermediate": "https://www.coursera.org/learn/data-science-methods",
        "advanced": "https://www.coursera.org/learn/advanced-data-science"
    },
    "sql": {
        "beginner": "https://www.coursera.org/learn/sql-for-data-science",
        "intermediate": "https://www.coursera.org/learn/intermediate-sql",
        "advanced": "https://www.coursera.org/learn/advanced-sql"
    },
    "git": {
        "beginner": "https://www.coursera.org/learn/git",
        "intermediate": "https://www.coursera.org/learn/git-intermediate",
        "advanced": "https://www.coursera.org/learn/git-advanced"
    },
    "docker": {
        "beginner": "https://www.coursera.org/learn/docker",
        "intermediate": "https://www.coursera.org/learn/docker-intermediate",
        "advanced": "https://www.coursera.org/learn/docker-advanced"
    },
    "kubernetes": {
        "beginner": "https://www.coursera.org/learn/kubernetes",
        "intermediate": "https://www.coursera.org/learn/kubernetes-intermediate",
        "advanced": "https://www.coursera.org/learn/kubernetes-advanced"
    },
    "javascript": {
        "beginner": "https://www.coursera.org/learn/javascript",
        "intermediate": "https://www.coursera.org/learn/javascript-intermediate",
        "advanced": "https://www.coursera.org/learn/javascript-advanced"
    },
    "typescript": {
        "beginner": "https://www.coursera.org/learn/typescript",
        "intermediate": "https://www.coursera.org/learn/typescript-intermediate",
        "advanced": "https://www.coursera.org/learn/typescript-advanced"
    },
    "react": {
        "beginner": "https://www.coursera.org/learn/react",
        "intermediate": "https://www.coursera.org/learn/react-intermediate",
        "advanced": "https://www.coursera.org/learn/react-advanced"
    },
    "vue": {
        "beginner": "https://www.coursera.org/learn/vue",
        "intermediate": "https://www.coursera.org/learn/vue-intermediate",
        "advanced": "https://www.coursera.org/learn/vue-advanced"
    },
    "html": {
        "beginner": "https://www.coursera.org/learn/html",
        "intermediate": "https://www.coursera.org/learn/html-intermediate",
        "advanced": "https://www.coursera.org/learn/html-advanced"
    },
    "css": {
        "beginner": "https://www.coursera.org/learn/css",
        "intermediate": "https://www.coursera.org/learn/css-intermediate",
        "advanced": "https://www.coursera.org/learn/css-advanced"
    },
    "rest api": {
        "beginner": "https://www.coursera.org/learn/rest-api",
        "intermediate": "https://www.coursera.org/learn/rest-api-intermediate",
        "advanced": "https://www.coursera.org/learn/rest-api-advanced"
    },
    "ci/cd": {
        "beginner": "https://www.coursera.org/learn/ci-cd",
        "intermediate": "https://www.coursera.org/learn/ci-cd-intermediate",
        "advanced": "https://www.coursera.org/learn/ci-cd-advanced"
    },
    "linux": {
        "beginner": "https://www.coursera.org/learn/linux",
        "intermediate": "https://www.coursera.org/learn/linux-intermediate",
        "advanced": "https://www.coursera.org/learn/linux-advanced"
    },
    "postgresql": {
        "beginner": "https://www.coursera.org/learn/postgresql",
        "intermediate": "https://www.coursera.org/learn/postgresql-intermediate",
        "advanced": "https://www.coursera.org/learn/postgresql-advanced"
    }
}

def recommend_courses(missing_skills: List[str], skill_level: str = "beginner") -> Dict[str, str]:
    """
    Рекомендует курсы для недостающих навыков.
    
    Args:
        missing_skills: Список недостающих навыков
        skill_level: Уровень курса (beginner/intermediate/advanced)
        
    Returns:
        Dict[str, str]: Словарь с рекомендациями курсов
    """
    try:
        logging.info(f"Рекомендация курсов для навыков: {missing_skills}")
        
        recommendations = {}
        for skill in missing_skills:
            skill = skill.lower()
            if skill in COURSES:
                if skill_level in COURSES[skill]:
                    recommendations[skill] = COURSES[skill][skill_level]
                else:
                    # Если уровень не найден, используем beginner
                    recommendations[skill] = COURSES[skill]["beginner"]
        
        logging.info(f"Найдено {len(recommendations)} рекомендаций")
        return recommendations
        
    except Exception as e:
        logging.error(f"Ошибка при рекомендации курсов: {str(e)}")
        return {}
