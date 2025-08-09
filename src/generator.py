import logging
import os

# Konfigurasi logging global dengan format standar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMGenerator:
    """
    Kelas untuk menghasilkan jawaban dari pertanyaan pengguna dengan menggunakan
    model Large Language Model (LLM) lokal melalui pustaka llama-cpp-python.
    """

    # Jumlah maksimum token yang dapat ditangani oleh model dalam satu input prompt
    MAX_CONTEXT_TOKENS = 2048

    # Jumlah maksimum token yang diperbolehkan dari dokumen (chunks) untuk membentuk prompt
    MAX_CHUNK_TOKENS = 150

    def __init__(self):
        """
        Konstruktor non-blok. Penundaan pemuatan model sampai pemanggilan pertama.
        """
        self.llm = None
        self._model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "model",
            "qwen1_5-1_8b-chat-q4_k_m.gguf"
        )
        # opsional: konfigurasi Groq via env var
        self._use_groq = bool(os.environ.get("GROQ_API_KEY"))

    def _ensure_model_loaded(self):
        if self.llm is not None:
            return True
        if not os.path.exists(self._model_path):
            logging.error(f"File model GGUF tidak ditemukan di: {self._model_path}. Mengandalkan Groq jika tersedia.")
            return False
        try:
            # impor malas untuk menghindari ketergantungan saat tidak digunakan
            from llama_cpp import Llama  # type: ignore
            self.llm = Llama(model_path=self._model_path, n_ctx=self.MAX_CONTEXT_TOKENS, verbose=False)
            # warmup singkat untuk memperkecil latensi panggilan pertama
            try:
                _ = self.llm.create_chat_completion(messages=[{"role": "user", "content": "Halo"}], max_tokens=1)
            except Exception:
                pass
            logging.info(f"Model Llama.cpp berhasil dimuat dengan context window {self.MAX_CONTEXT_TOKENS}.")
            return True
        except Exception as e:
            logging.error(f"Gagal memuat model Llama.cpp: {e}")
            return False

    def _create_prompt(self, query, retrieved_chunks):
        """
        Membuat prompt input untuk LLM dari query pengguna dan dokumen yang relevan.

        Parameters:
            query (str): Pertanyaan dari pengguna.
            retrieved_chunks (List[str]): List dokumen atau paragraf yang relevan.

        Returns:
            str: Prompt lengkap yang akan dikirim ke model LLM.
        """
        # Template dasar untuk sistem prompt
        prompt_template = (
            "Anda adalah asisten AI yang ahli dalam Peraturan Daerah Kota Bandung "
            "tentang Pengelolaan Sampah. Tugas Anda adalah memberikan jawaban yang singkat, "
            "sopan, dan berbasis fakta dari dokumen yang disediakan.\n\n"
            "Dokumen terkait:\n"
            "{chunks}\n\n"
            "Berdasarkan dokumen di atas, jawab pertanyaan pengguna berikut:\n"
            "Pertanyaan: {query}\n"
            "Jawaban:"
        )

        # Bangun konten dokumen yang dibatasi jumlah token-nya
        chunks_text_list = []
        current_token_count = 0
        for chunk in retrieved_chunks:
            chunk_tokens = len(chunk.split())
            if current_token_count + chunk_tokens <= self.MAX_CHUNK_TOKENS:
                chunks_text_list.append(chunk)
                current_token_count += chunk_tokens
            else:
                break  # Hentikan jika melebihi batas

        chunks_text = "\n---\n".join(chunks_text_list)
        prompt = prompt_template.format(chunks=chunks_text, query=query)

        return prompt

    def _groq_stream_completion(self, prompt_content):
        """Opsional: streaming via Groq jika tersedia, fallback ke model lokal jika gagal."""
        try:
            from groq import Groq
            client = Groq()
            stream = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Anda adalah asisten AI yang ahli dalam Peraturan Daerah Kota Bandung tentang Pengelolaan Sampah."},
                    {"role": "user", "content": prompt_content}
                ],
                model=os.environ.get("GROQ_MODEL", "llama3-8b-8192"),
                temperature=0.7,
                max_tokens=256,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta or {}
                if "content" in delta and delta["content"]:
                    yield delta["content"]
        except Exception as e:
            logging.warning(f"Groq streaming gagal atau tidak tersedia: {e}")
            # fallback: tidak yield apa-apa, caller akan coba lokal
            return

    def _extractive_fallback(self, query, retrieved_chunks):
        """Fallback sederhana saat tidak ada model: kembalikan ringkasan ekstraktif dari chunk teratas."""
        try:
            # Ambil 2-3 kalimat awal dari chunk pertama sebagai jawaban ringkas
            import re
            top_text = (retrieved_chunks[0] if retrieved_chunks else "")[:1000]
            sentences = re.split(r"(?<=[.!?])\s+", top_text)
            answer = " ".join(sentences[:3])
            if len(answer) < 20:
                # jika terlalu pendek, tambahkan sebagian chunk kedua
                extra = (retrieved_chunks[1] if len(retrieved_chunks) > 1 else "")[:300]
                answer = (answer + " " + extra).strip()
            prefix = "(Jawaban ringkas berdasarkan dokumen terkait, model sedang tidak tersedia) "
            return (prefix + answer)[:512]
        except Exception:
            return "Maaf, sementara model tidak tersedia. Silakan coba lagi nanti."

    def generate_answer(self, query, retrieved_chunks):
        """
        Menghasilkan jawaban; melakukan lazy-load model, dan menggunakan streaming bila memungkinkan.
        """
        # Validasi awal sebelum memproses prompt
        if not query.strip():
            return "Pertanyaan tidak boleh kosong. Mohon masukkan pertanyaan Anda."

        if not retrieved_chunks:
            return "Maaf, saya tidak dapat menemukan informasi yang relevan dalam dokumen."

        # Bangun prompt dari query dan chunks
        prompt_content = self._create_prompt(query, retrieved_chunks)

        # Validasi panjang prompt sebelum dikirim
        if len(prompt_content.split()) > self.MAX_CONTEXT_TOKENS:
            logging.error("Prompt masih terlalu panjang setelah pemotongan. Silakan kurangi ukuran chunks lebih lanjut.")
            return "Maaf, prompt yang dihasilkan terlalu panjang untuk model. Silakan coba pertanyaan yang lebih ringkas."

        # Prioritas: Groq dengan streaming jika tersedia
        if self._use_groq:
            collected = []
            for token in self._groq_stream_completion(prompt_content):
                collected.append(token)
            if collected:
                return "".join(collected)
            # jika Groq gagal, lanjut ke lokal

        # Lokal: pastikan model termuat dan gunakan streaming jika didukung
        if not self._ensure_model_loaded():
            # fallback heuristik agar tetap ada jawaban offline
            return self._extractive_fallback(query, retrieved_chunks)

        logging.info("Prompt berhasil dibuat. Menghasilkan jawaban dengan model LLM...")

        try:
            # llama_cpp mendukung streaming token
            stream = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Anda adalah asisten AI yang ahli dalam Peraturan Daerah Kota Bandung tentang Pengelolaan Sampah."},
                    {"role": "user", "content": prompt_content}
                ],
                stop=["Jawaban:", "###"],  # Pemicu untuk menghentikan output
                max_tokens=256,             # Batas panjang jawaban
                temperature=0.7,             # Kontrol variasi jawaban
                stream=True,
            )
            output_parts = []
            for chunk in stream:
                if "choices" in chunk and chunk["choices"] and "delta" in chunk["choices"][0]:
                    delta = chunk["choices"][0]["delta"]
                    if isinstance(delta, dict) and "content" in delta and delta["content"]:
                        output_parts.append(delta["content"])
            if output_parts:
                return "".join(output_parts)
            # fallback non-streaming
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Anda adalah asisten AI yang ahli dalam Peraturan Daerah Kota Bandung tentang Pengelolaan Sampah."},
                    {"role": "user", "content": prompt_content}
                ],
                stop=["Jawaban:", "###"],
                max_tokens=256,
                temperature=0.7,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Gagal menghasilkan respons dari model: {e}")
            return self._extractive_fallback(query, retrieved_chunks)