from typing import List, Dict, Tuple
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('skill_comparison.log'),
        logging.StreamHandler()
    ]
)

# Соответствие уровней
LEVEL_MAPPING = {
    "beginner": 1,      # Базовый
    "intermediate": 2,  # Продвинутый
    "advanced": 3,      # Экспертный
    "unknown": 1        # По умолчанию считаем базовым
}

def normalize_skill(skill: str) -> str:
    """Нормализует название навыка для сравнения."""
    return skill.lower().strip()

def get_level_score(candidate_level: str, required_level: int) -> float:
    """
    Рассчитывает оценку соответствия уровня навыка.
    
    Args:
        candidate_level: Уровень кандидата (beginner/intermediate/advanced)
        required_level: Требуемый уровень (1/2/3)
        
    Returns:
        float: Оценка соответствия от 0 до 1
    """
    candidate_level_num = LEVEL_MAPPING.get(candidate_level, 1)
    
    if candidate_level_num >= required_level:
        return 1.0  # Полное соответствие
    elif candidate_level_num == required_level - 1:
        return 0.5  # Частичное соответствие
    else:
        return 0.0  # Несоответствие

def compare_skills(candidate_skills: List[Dict[str, str]], required_skills: Dict[str, int]) -> Tuple[float, List[str], List[Dict]]:
    """
    Сравнивает навыки кандидата с требуемыми навыками.
    
    Args:
        candidate_skills: Список навыков кандидата с уровнями
        required_skills: Словарь требуемых навыков с уровнями
        
    Returns:
        Tuple[float, List[str], List[Dict]]: 
            - Оценка соответствия (0-1)
            - Список недостающих навыков
            - Список совпавших навыков с информацией о соответствии
    """
    try:
        logging.info("Начало сравнения навыков")
        
        if not candidate_skills or not required_skills:
            return 0.0, list(required_skills.keys()), []
        
        # Создаем словарь навыков кандидата для быстрого поиска
        candidate_skill_dict = {
            normalize_skill(skill["skill"]): skill
            for skill in candidate_skills
        }
        
        # Инициализируем результаты
        total_score = 0.0
        missing_skills = []
        matched_skills = []
        
        # Проходим по требуемым навыкам
        for skill_name, required_level in required_skills.items():
            norm_skill = normalize_skill(skill_name)
            
            if norm_skill in candidate_skill_dict:
                candidate_skill = candidate_skill_dict[norm_skill]
                level_score = get_level_score(candidate_skill["level"], required_level)
                
                # Учитываем уверенность в навыке
                confidence = float(candidate_skill.get("confidence", 0.5))
                final_score = level_score * confidence
                
                # Определяем тип совпадения
                match_level = "full" if level_score == 1.0 else "partial" if level_score > 0 else "none"
                
                matched_skills.append({
                    "skill": skill_name,
                    "level": candidate_skill["level"],
                    "required_level": required_level,
                    "match_level": match_level,
                    "score": final_score
                })
                
                total_score += final_score
            else:
                missing_skills.append(skill_name)
        
        # Нормализуем общую оценку
        final_score = total_score / len(required_skills) if required_skills else 0.0
        
        logging.info(f"Оценка соответствия: {final_score:.2f}")
        logging.info(f"Найдено совпадений: {len(matched_skills)}")
        logging.info(f"Недостающие навыки: {missing_skills}")
        
        return final_score, missing_skills, matched_skills
        
    except Exception as e:
        logging.error(f"Ошибка при сравнении навыков: {str(e)}")
        return 0.0, list(required_skills.keys()), []
