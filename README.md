# AI Skill Matcher

Streamlit-приложение для анализа соответствия навыков в резюме выбранному профилю.

## 📁 Содержимое проекта

- `app.py` — основной интерфейс приложения
- `resume_parser.py` — извлечение текста из PDF, DOCX, TXT
- `extract_skills.py` — извлечение ключевых навыков
- `compare_skills.py` — сравнение с профилем
- `recommend_courses.py` — подбор курсов
- `styles/main.css` — кастомные стили
- `requirements.txt` — список зависимостей

## 🚀 Запуск проекта

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Запустите приложение:
```bash
streamlit run app.py
```

3. Перейдите в браузер по адресу:
```
http://localhost:8501
```

## ⚠️ Зависимости

Для PDF используется PyMuPDF (fitz).  
Для DOCX — `docx2txt` (устойчивее `python-docx`).

## 🌐 Поддержка форматов

- ✅ .pdf
- ✅ .docx
- ✅ .txt

## 📌 Требования

- Python 3.9–3.11