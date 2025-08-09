import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    """
    Kelas untuk memuat data dokumen yang telah diproses dan mengambil bagian (chunk) 
    yang paling relevan terhadap query menggunakan TF-IDF dan cosine similarity.
    """

    def __init__(self, data_path="perda_data.pkl"):
        """
        Inisialisasi DocumentRetriever.

        Args:
            data_path (str): Path ke file pickle yang berisi data dokumen, vectorizer, dan TF-IDF matrix.
        """
        self.data_path = data_path
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        # cache untuk hasil retrieval berulang
        from collections import OrderedDict
        self._result_cache = OrderedDict()
        self._cache_maxsize = 128
        self._load_data()

    def __str__(self):
        """
        Representasi string dari objek.

        Returns:
            str: Ringkasan jumlah chunks yang dimuat.
        """
        return f"<DocumentRetriever | chunks: {len(self.chunks)}>"

    def _load_data(self):
        """
        Memuat data dari file pickle yang telah diproses sebelumnya.
        File ini harus berisi kunci 'chunks', 'vectorizer', dan 'tfidf_matrix'.
        Jika salah satu hilang, maka data dianggap tidak valid.
        """
        if os.path.exists(self.data_path):
            try:
                data = joblib.load(self.data_path)
                self.chunks = data.get('chunks', [])
                self.vectorizer = data.get('vectorizer')
                self.tfidf_matrix = data.get('tfidf_matrix')
                if not self.chunks or self.vectorizer is None:
                    logging.error("Data yang dimuat tidak lengkap. Pastikan perda_data.pkl valid.")
                    self.chunks = []  # Reset untuk mencegah penggunaan data tidak valid
                    return
                # Validasi kompatibilitas/fit; rebuild jika perlu
                needs_rebuild = False
                try:
                    _ = self.vectorizer.transform(["uji"])  # akan gagal jika tidak fitted / versi tidak cocok
                    if not hasattr(self.vectorizer, "idf_"):
                        needs_rebuild = True
                except Exception:
                    needs_rebuild = True
                if needs_rebuild or self.tfidf_matrix is None:
                    logging.warning("Vectorizer tidak terpasang dengan benar atau versi tidak cocok. Membangun ulang indeks TF-IDF dari chunks...")
                    import numpy as np
                    self.vectorizer = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
                logging.info(f"Data retriever berhasil dimuat dari pickle gabungan. Total chunks: {len(self.chunks)}")
                return
            except Exception as e:
                logging.error(f"Gagal memuat data dari {self.data_path}: {e}")
                self.chunks = []
                # lanjut coba load format terpisah
        # Coba format artefak terpisah: <base>_vectorizer.pkl dan <base>_tfidf.npz
        try:
            from scipy.sparse import load_npz
            base, _ = os.path.splitext(self.data_path)
            vec_path = base + "_vectorizer.pkl"
            mat_path = base + "_tfidf.npz"
            if not (os.path.exists(vec_path) and os.path.exists(mat_path)):
                logging.error(f"Artefak retriever tidak ditemukan: {vec_path} atau {mat_path}. Jalankan perda_processor.py.")
                return
            self.vectorizer = joblib.load(vec_path)
            self.tfidf_matrix = load_npz(mat_path)
            # muat chunks
            chunks_path = base + "_chunks.pkl"
            if os.path.exists(chunks_path):
                self.chunks = joblib.load(chunks_path)
            else:
                # fallback: jika chunks tidak disimpan terpisah, coba baca dari pickle gabungan jika ada
                logging.warning("File chunks terpisah tidak ditemukan. Pastikan menyimpan chunks ke <base>_chunks.pkl untuk performa optimal.")
                self.chunks = []
            if not self.chunks:
                logging.error("Data chunks kosong. Tidak dapat melakukan retrieval.")
                return
            logging.info(f"Data retriever berhasil dimuat dari artefak terpisah. Total chunks: {len(self.chunks)}")
        except Exception as e:
            logging.error(f"Gagal memuat artefak terpisah: {e}")
            self.chunks = []

    def retrieve_chunks(self, query, top_k=3):
        """
        Mengambil potongan dokumen (chunks) yang paling relevan terhadap query yang diberikan
        menggunakan cosine similarity terhadap representasi TF-IDF.

        Args:
            query (str): Pertanyaan atau masukan dari pengguna.
            top_k (int): Jumlah top hasil paling relevan yang ingin dikembalikan.

        Returns:
            list[str]: Daftar chunks teks yang paling relevan terhadap query.
        """
        if not self.chunks or self.vectorizer is None:
            logging.warning("Retriever tidak siap. Kembalikan array kosong.")
            return []
        if not query.strip():
            logging.info("Query kosong. Tidak ada retrieval yang dilakukan.")
            return []
        # cache hit
        cache_key = (query, int(top_k))
        if cache_key in self._result_cache:
            # pindahkan ke belakang (MRU)
            result = self._result_cache.pop(cache_key)
            self._result_cache[cache_key] = result
            return result
        # kandidat indeks (jika ada filter keyword-to-pasal, biarkan tetap bekerja di luar blok ini)
        try:
            import numpy as np
            from sklearn.metrics.pairwise import linear_kernel
            query_vector = self.vectorizer.transform([query])
            # gunakan linear kernel yang efisien untuk TF-IDF
            scores = linear_kernel(query_vector, self.tfidf_matrix).ravel()
            # ambil top_k tanpa sort penuh
            if top_k <= 0:
                top_k = 1
            top_k = min(top_k, scores.shape[0])
            top_k_indices = np.argpartition(scores, -top_k)[-top_k:]
            # urutkan hanya kandidat top_k secara descending
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
            relevant_chunks = [self.chunks[i] for i in top_k_indices]
        except Exception as e:
            logging.error(f"Kesalahan saat menghitung kesamaan: {e}. Menggunakan metode cadangan.")
            # fallback ke metode lama agar akurasi tetap terjaga jika terjadi error
            query_vector = self.vectorizer.transform([query])
            from sklearn.metrics.pairwise import cosine_similarity
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_k_indices = cosine_similarities.argsort()[-top_k:][::-1]
            relevant_chunks = [self.chunks[i] for i in top_k_indices]
        logging.info(f"Ditemukan {len(relevant_chunks)} chunks relevan untuk query.")
        # simpan ke cache LRU kecil
        self._result_cache[cache_key] = relevant_chunks
        if len(self._result_cache) > self._cache_maxsize:
            # hapus item tertua
            self._result_cache.popitem(last=False)
        return relevant_chunks

# Contoh penggunaan
if __name__ == "__main__":
    retriever = DocumentRetriever()
    print(retriever)
    
    test_query = "Bagaimana cara membuang sachet kopi?"
    relevant_docs = retriever.retrieve_chunks(test_query, top_k=3)
    
    print("\n--- Hasil Retrieval ---")
    print(f"Query: {test_query}")
    for i, chunk in enumerate(relevant_docs):
        print(f"\nChunk {i+1}:\n{chunk[:250]}...")