import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    """Konfigurasi aplikasi, diatur melalui environment variables."""
    # API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # LLM Parameters
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    # LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

    # Prompting
    SYSTEM_PROMPT = (
        "Anda adalah seorang profesional di bidang hukum yang sangat menguasai "
        "Peraturan Nasional tentang Pengelolaan Sampah. "
        "Tugas Anda adalah: "
        "1. Memberikan jawaban yang akurat, singkat, dan langsung pada pokok permasalahan. "
        "2. Jawablah hanya berdasarkan informasi dari konteks dokumen yang diberikan. "
        "3. Selalu kutip nomor pasal, ayat, dan/atau sumber spesifik (misalnya, [sumber: X]) "
        "   untuk setiap informasi yang Anda berikan. "
        "4. Jika informasi tidak ada di dalam konteks, nyatakan dengan sopan bahwa Anda tidak "
        "   dapat menemukan informasi tersebut. "
        "5. Gunakan format yang jelas dan mudah dibaca (misalnya, daftar, poin-poin penting)."
    )