import os
import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

# === 1. НАСТРОЙКА НЕЙРОСЕТЕЙ ===

# Получаем ключ доступа Hugging Face из секретных переменных окружения
HF_TOKEN = os.environ.get("HF_TOKEN")

# Инициализируем клиента для общения с большой языковой моделью (LLM)
# Мы используем Qwen 2.5 1.5B — она умная, быстрая и отлично понимает инструкции
client = InferenceClient(model="Qwen/Qwen2.5-1.5B-Instruct", token=HF_TOKEN)

# Загружаем компактную модель эмбеддингов
# Эмбеддинги превращают текст в наборы чисел (векторы), чтобы компьютер мог сравнивать их по смыслу
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# === 2. ПОДГОТОВКА БАЗЫ ЗНАНИЙ ===

def prepare_knowledge_base():
    """Загружает книгу правил и превращает её в цифровой поиск."""
    path = "fate-core.md"
    
    if not os.path.exists(path):
        # Если файл правил вдруг исчезнет, приложение выдаст ошибку вместо падения
        return ["Rulebook file not found."], np.zeros((1, 384))
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Нарезаем текст на смысловые куски (абзацы), чтобы модели было проще их «переварить»
    # Оставляем только фрагменты длиннее 50 символов, чтобы отсечь мусор
    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 50]
    
    # Переводим каждый кусок текста в вектор (набор чисел)
    # Теперь мы сможем искать правила не по словам, а по смыслу
    embeddings = embed_model.encode(chunks)
    return chunks, embeddings

# Выполняем индексацию один раз при запуске приложения
CHUNKS, CHUNK_EMBEDS = prepare_knowledge_base()

# === 3. ЛОГИКА RAG (Retrieval-Augmented Generation) ===

def ask_fate(question):
    """Находит нужные правила и формирует ответ с помощью ИИ."""
    if not question.strip():
        return "Пожалуйста, введите ваш вопрос."
    
    try:
        # 1. ПОИСК: Превращаем вопрос пользователя в вектор
        q_embed = embed_model.encode([question])
        
        # 2. СРАВНЕНИЕ: Считаем сходство вопроса со всеми фрагментами правил
        # Используем матричное умножение для мгновенного поиска похожих смыслов
        scores = np.dot(CHUNK_EMBEDS, q_embed.T).flatten()
        
        # Берем 3 самых подходящих фрагмента текста
        top_idx = np.argsort(scores)[-3:][::-1]
        context = "\n\n---\n\n".join([CHUNKS[i] for i in top_idx])
        
        # 3. ГЕНЕРАЦИЯ: Формируем задание для языковой модели
        # Передаем найденный текст (контекст) и сам вопрос
        messages = [
            {
                "role": "system", 
                "content": "Ты — эксперт по настольной игре Fate Core. Отвечай ТОЛЬКО на основе предоставленного текста правил. Если ответа нет в тексте — честно скажи об этом."
            },
            {
                "role": "user", 
                "content": f"ВЫДЕРЖКИ ИЗ ПРАВИЛ:\n{context}\n\nВОПРОС: {question}"
            }
        ]
        
        # Отправляем запрос в облачную нейросеть
        response = client.chat_completion(
            messages=messages,
            max_tokens=500,
            temperature=0.1 # Минимальная температура делает ответы точными и строгими
        )
        
        # Возвращаем итоговый текст ответа
        return response.choices[0].message.content

    except Exception as e:
        return f"Произошла техническая ошибка: {str(e)}"

# === 4. СОЗДАНИЕ ВЕБ-ИНТЕРФЕЙСА ===

# Используем библиотеку Gradio для создания удобного окна чата
demo = gr.Interface(
    fn=ask_fate,
    inputs=gr.Textbox(
        label="Задайте вопрос Оракулу Fate", 
        placeholder="Например: Как работают аспекты?"
    ),
    outputs=gr.Markdown(), # Markdown позволяет красиво отображать списки и жирный текст
    title="🎲 Fate Core Oracle",
    description="Интеллектуальный помощник по правилам Fate Core. Система находит нужный раздел в книге и объясняет его своими словами.",
    theme=gr.themes.Soft() # Приятная мягкая тема оформления
)

if __name__ == "__main__":
    # Запускаем сервер
    demo.launch()
