import re
import logging
from typing import List, Dict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Определяем конкретные технические навыки
SKILLS = {
    # Языки программирования и основные библиотеки
    'python': {
        'category': 'Programming Languages',
        'patterns': [
            r'(?:^|[^\w])(python|python3)(?:[^\w]|$)',
            r'(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer).{0,30}python',
            r'python.{0,30}(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer)',
        ]
    },
    'java': {
        'category': 'Programming Languages',
        'patterns': [
            r'(?:^|[^\w])(java\s*(?:8|11|17)?|spring\s*boot)(?:[^\w]|$)',
            r'(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer).{0,30}java',
            r'java.{0,30}(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer)',
        ]
    },
    'javascript': {
        'category': 'Programming Languages',
        'patterns': [
            r'(?:^|[^\w])(javascript|js|node\.js|nodejs|typescript|ts)(?:[^\w]|$)',
            r'(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer).{0,30}(?:javascript|js)',
            r'(?:javascript|js).{0,30}(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer)',
        ]
    },
    'c++': {
        'category': 'Programming Languages',
        'patterns': [
            r'(?:^|[^\w])(c\+\+|cpp|с\+\+)(?:[^\w]|$)',
            r'(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer).{0,30}c\+\+',
            r'c\+\+.{0,30}(?:программ(?:ист|ирование)|разработ[кч]|engineer|developer)',
        ]
    },

    # ML & DL Frameworks and Tools
    'machine_learning': {
        'category': 'Machine Learning',
        'patterns': [
            r'(?:^|[^\w])(machine\s*learning|ml|машинное\s*обучение)(?:[^\w]|$)',
            r'(?:ai|artificial\s*intelligence|искусственный\s*интеллект)',
            r'(?:ml\s*engineer|ai\s*engineer|ml\s*specialist)',
        ]
    },
    'scikit_learn': {
        'category': 'Machine Learning',
        'patterns': [
            r'(?:^|[^\w])(scikit[\s-]*learn|sklearn)(?:[^\w]|$)',
            r'(?:ml|machine\s*learning).{0,30}(?:scikit|sklearn)',
        ]
    },
    'xgboost': {
        'category': 'Machine Learning',
        'patterns': [
            r'(?:^|[^\w])(xgboost|gradient\s*boosting)(?:[^\w]|$)',
        ]
    },
    'catboost': {
        'category': 'Machine Learning',
        'patterns': [
            r'(?:^|[^\w])(catboost)(?:[^\w]|$)',
        ]
    },
    'tensorflow': {
        'category': 'Machine Learning',
        'patterns': [
            r'(?:^|[^\w])(tensorflow|tf)(?:[^\w]|$)',
            r'(?:deep\s*learning|ml|machine\s*learning|ai).{0,30}tensorflow',
        ]
    },
    'pytorch': {
        'category': 'Machine Learning',
        'patterns': [
            r'(?:^|[^\w])(pytorch|torch)(?:[^\w]|$)',
            r'(?:deep\s*learning|ml|machine\s*learning|ai).{0,30}pytorch',
        ]
    },
    'transformers': {
        'category': 'Machine Learning',
        'patterns': [
            r'(?:^|[^\w])(transformers?|bert|gpt|keybert|llm)(?:[^\w]|$)',
            r'(?:hugging\s*face|🤗)',
        ]
    },

    # NLP Tools
    'nlp': {
        'category': 'Natural Language Processing',
        'patterns': [
            r'(?:^|[^\w])(nlp|natural\s*language\s*processing)(?:[^\w]|$)',
            r'(?:text\s*analysis|language\s*model)',
        ]
    },
    'bert': {
        'category': 'Natural Language Processing',
        'patterns': [
            r'(?:^|[^\w])(bert|roberta|distilbert)(?:[^\w]|$)',
        ]
    },
    'gpt': {
        'category': 'Natural Language Processing',
        'patterns': [
            r'(?:^|[^\w])(gpt|gpt-[23]|chatgpt)(?:[^\w]|$)',
        ]
    },
    'spacy': {
        'category': 'Natural Language Processing',
        'patterns': [
            r'(?:^|[^\w])(spacy|spaCy)(?:[^\w]|$)',
        ]
    },
    'rag': {
        'category': 'Natural Language Processing',
        'patterns': [
            r'(?:^|[^\w])(rag|retrieval[\s-]*augmented\s*generation)(?:[^\w]|$)',
            r'rag\s*pipelines?',
        ]
    },

    # MLOps & Infrastructure
    'mlops': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(mlops|ml\s*ops|mlflow|dvc)(?:[^\w]|$)',
            r'(?:model\s*deployment|model\s*monitoring|model\s*serving)',
        ]
    },
    'mlflow': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(mlflow)(?:[^\w]|$)',
        ]
    },
    'dvc': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(dvc|data\s*version\s*control)(?:[^\w]|$)',
        ]
    },
    'docker': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(docker|docker\-compose|контейнеризац)(?:[^\w]|$)',
            r'(?:container|deployment).{0,30}docker',
        ]
    },
    'kubernetes': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(kubernetes|k8s|k3s)(?:[^\w]|$)',
            r'(?:orchestration|deployment).{0,30}(?:kubernetes|k8s)',
        ]
    },
    'fastapi': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(fastapi|fast\s*api)(?:[^\w]|$)',
        ]
    },
    'airflow': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(airflow|apache\s*airflow)(?:[^\w]|$)',
        ]
    },
    'ci_cd': {
        'category': 'MLOps',
        'patterns': [
            r'(?:^|[^\w])(ci/cd|cicd|continuous\s*integration|continuous\s*deployment)(?:[^\w]|$)',
        ]
    },

    # Cloud Platforms
    'aws': {
        'category': 'Cloud',
        'patterns': [
            r'(?:^|[^\w])(aws|amazon\s*web\s*services|sagemaker|lambda|ec2|s3)(?:[^\w]|$)',
        ]
    },
    'gcp': {
        'category': 'Cloud',
        'patterns': [
            r'(?:^|[^\w])(gcp|google\s*cloud|vertex\s*ai)(?:[^\w]|$)',
        ]
    },

    # Data Tools
    'pandas': {
        'category': 'Data Tools',
        'patterns': [
            r'(?:^|[^\w])(pandas|pd\.)(?:[^\w]|$)',
        ]
    },
    'numpy': {
        'category': 'Data Tools',
        'patterns': [
            r'(?:^|[^\w])(numpy|np\.)(?:[^\w]|$)',
        ]
    },
    'spark': {
        'category': 'Data Tools',
        'patterns': [
            r'(?:^|[^\w])(spark|pyspark|apache\s*spark)(?:[^\w]|$)',
        ]
    },
    'sql': {
        'category': 'Data Tools',
        'patterns': [
            r'(?:^|[^\w])(sql|mysql|postgresql|oracle)(?:[^\w]|$)',
        ]
    },
    'nosql': {
        'category': 'Data Tools',
        'patterns': [
            r'(?:^|[^\w])(nosql|mongodb|cassandra|redis)(?:[^\w]|$)',
        ]
    },
    'bi_tools': {
        'category': 'Data Tools',
        'patterns': [
            r'(?:^|[^\w])(bi\s*tools|tableau|power\s*bi|looker)(?:[^\w]|$)',
        ]
    }
}

def determine_skill_level(context: str, full_text: str = "") -> str:
    """Определяет уровень навыка на основе контекста и полного текста резюме."""
    context = (context + " " + full_text).lower()
    
    # Паттерны для определения уровня
    expert_patterns = [
        # Должности и роли
        r'(?:senior|ведущий|главный|lead)\s*(?:developer|engineer|разработчик|программист|specialist|специалист)',
        r'(?:team\s*lead|тим\s*лид|архитектор|architect)',
        r'(?:head\s*of|director\s*of|chief)',
        
        # Опыт работы
        r'(?:seasoned|experienced)\s*(?:specialist|engineer|developer)',
        r'опыт.{1,20}(?:более|over|>\s*).{1,10}(?:5|6|7|8|9|10).{1,10}(?:лет|years)',
        r'[5-9](?:\+|\s*\+)?\s*(?:years?|лет|года)\s*(?:опыта|experience)',
        r'(?:10|11|12|13|14|15)\+?\s*(?:years?|лет|года)\s*(?:опыта|experience)',
        
        # Уровень экспертизы
        r'эксперт|expert',
        r'большой опыт|extensive experience',
        r'advanced knowledge|продвинутые знания',
        r'глубокие (?:знания|познания)',
        r'уверенное владение',
        r'отличное знание|excellent knowledge',
        
        # Лидерство и достижения
        r'led\s*(?:a\s*team|development|implementation)',
        r'architected|designed|implemented',
        r'improved\s*(?:performance|efficiency|scalability).{1,20}\d+%',
    ]
    
    intermediate_patterns = [
        r'(?:middle|миддл)',
        r'опыт.{1,20}(?:более|over|>\s*).{1,10}[2-4].{1,10}(?:лет|years)',
        r'[2-4](?:\+|\s*\+)?\s*(?:years?|лет|года)\s*(?:опыта|experience)',
        r'(?:хорошее|good)\s*(?:знание|knowledge)',
        r'(?:уверенное|confident)\s*(?:знание|knowledge)',
        r'experienced|опытный',
        r'proficient|уверенный',
        r'практический опыт|practical experience',
    ]
    
    # Проверяем паттерны уровней
    for pattern in expert_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return 'expert'
            
    for pattern in intermediate_patterns:
        if re.search(pattern, context, re.IGNORECASE):
            return 'intermediate'
            
    return 'beginner'

def extract_skills(text: str) -> List[Dict]:
    """
    Извлекает технические навыки из текста, используя строгие регулярные выражения.
    
    Args:
        text: Текст резюме
        
    Returns:
        List[Dict]: Список навыков с их уровнями и уверенностью определения
    """
    try:
        if not text:
            return []
            
        # Очищаем текст
        text = text.lower()
        # Заменяем переносы строк пробелами
        text = re.sub(r'\s+', ' ', text)
        
        # Словарь для хранения найденных навыков
        found_skills = {}
        
        # Ищем навыки по паттернам
        for skill_name, skill_info in SKILLS.items():
            for pattern in skill_info['patterns']:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in matches:
                    # Получаем контекст (100 символов до и после совпадения)
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]
                    
                    # Определяем уровень на основе контекста и полного текста
                    level = determine_skill_level(context, text)
                    
                    # Вычисляем уверенность
                    confidence = 0.8  # Базовая уверенность
                    
                    # Повышаем уверенность в зависимости от контекста
                    if re.search(r'опыт.{1,20}(?:работы|разработки|использования).{1,30}' + re.escape(skill_name), context, re.IGNORECASE):
                        confidence += 0.1
                    if re.search(r'\d+\s*(?:years?|лет|года)', context, re.IGNORECASE):
                        confidence += 0.1
                    if re.search(r'(?:advanced|expert|senior|ведущий|главный|lead)', context, re.IGNORECASE):
                        confidence += 0.1
                    
                    # Если навык упоминается в контексте обучения или планов, уменьшаем уверенность
                    if re.search(r'(?:studying|learning|курсы|обучение|изучаю|изучение|планирую|plan\s+to)', context, re.IGNORECASE):
                        confidence -= 0.2
                        
                    # Сохраняем навык с максимальной уверенностью
                    if skill_name not in found_skills or found_skills[skill_name]['confidence'] < confidence:
                        found_skills[skill_name] = {
                            'skill': skill_name,
                            'level': level,
                            'confidence': confidence,
                            'category': skill_info['category']
                        }

        # Преобразуем результаты в список и сортируем по уверенности
        results = list(found_skills.values())
        # Фильтруем навыки с низкой уверенностью
        results = [skill for skill in results if skill['confidence'] >= 0.7]
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    except Exception as e:
        logging.error(f"Ошибка при извлечении навыков: {str(e)}")
        return []
