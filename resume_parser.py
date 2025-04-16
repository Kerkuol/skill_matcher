import fitz  # PyMuPDF
import docx2txt
import io
import logging
import os
from typing import Union
import tempfile

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resume_parser.log'),
        logging.StreamHandler()
    ]
)

def extract_text_from_pdf(file_content: bytes) -> str:
    """Извлекает текст из PDF файла."""
    try:
        if not file_content:
            raise ValueError("Получено пустое содержимое файла")
            
        text = ""
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Ошибка при извлечении текста из PDF: {str(e)}")
        raise

def extract_text_from_docx(file_content: bytes) -> str:
    """Извлекает текст из DOCX файла."""
    tmp_file_path = None
    try:
        if not file_content:
            raise ValueError("Получено пустое содержимое файла")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        text = docx2txt.process(tmp_file_path)
        if not text:
            raise ValueError("Не удалось извлечь текст из файла")
            
        return text.strip()
    except Exception as e:
        logging.error(f"Ошибка при извлечении текста из DOCX: {str(e)}")
        raise
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logging.warning(f"Не удалось удалить временный файл {tmp_file_path}: {str(e)}")

def extract_text_from_txt(file_content: bytes) -> str:
    """Извлекает текст из TXT файла."""
    try:
        if not file_content:
            raise ValueError("Получено пустое содержимое файла")
            
        try:
            text = file_content.decode('utf-8').strip()
        except UnicodeDecodeError:
            text = file_content.decode('cp1251', errors='ignore').strip()
            
        if not text:
            raise ValueError("Не удалось извлечь текст из файла")
            
        return text
    except Exception as e:
        logging.error(f"Ошибка при извлечении текста из TXT: {str(e)}")
        raise

def parse_resume(uploaded_file) -> str:
    """
    Парсит резюме из загруженного файла и возвращает извлеченный текст.
    
    Args:
        uploaded_file: Загруженный файл из streamlit
        
    Returns:
        str: Извлеченный текст или сообщение об ошибке
    """
    try:
        if uploaded_file is None:
            logging.error("Файл не загружен")
            return "[❌ Файл не загружен]"
        
        # Проверяем размер файла
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        file_size = len(uploaded_file.getvalue())
        if file_size > MAX_FILE_SIZE:
            logging.warning(f"Файл слишком большой: {file_size} bytes")
            return f"[⚠️ Файл слишком большой. Максимальный размер: {MAX_FILE_SIZE/1024/1024:.1f}MB]"
        elif file_size == 0:
            logging.warning("Загружен пустой файл")
            return "[⚠️ Файл пуст]"
            
        name = uploaded_file.name.lower()
        file_content = uploaded_file.getvalue()
        
        # Проверяем формат файла
        supported_formats = {".pdf", ".docx", ".txt"}
        file_ext = os.path.splitext(name)[1]
        if file_ext not in supported_formats:
            logging.warning(f"Неподдерживаемый формат файла: {name}")
            return f"[⚠️ Неподдерживаемый формат файла. Поддерживаемые форматы: {', '.join(supported_formats)}]"
        
        logging.info(f"Начало обработки файла: {name}")
        
        # Извлекаем текст в зависимости от формата
        try:
            if file_ext == ".pdf":
                text = extract_text_from_pdf(file_content)
            elif file_ext == ".docx":
                text = extract_text_from_docx(file_content)
            elif file_ext == ".txt":
                text = extract_text_from_txt(file_content)
            else:
                return "[⚠️ Неподдерживаемый формат файла]"
                
            if not text.strip():
                logging.warning(f"Не удалось извлечь текст из файла: {name}")
                return "[⚠️ Не удалось извлечь текст из файла]"
                
            logging.info(f"Успешно обработан файл: {name}")
            return text
            
        except Exception as e:
            logging.error(f"Ошибка при извлечении текста из файла {name}: {str(e)}")
            return f"[❌ Ошибка обработки: {str(e)}]"
        
    except Exception as e:
        logging.error(f"Ошибка при обработке файла {uploaded_file.name if uploaded_file else 'Unknown'}: {str(e)}")
        return f"[❌ Ошибка обработки: {str(e)}]"
