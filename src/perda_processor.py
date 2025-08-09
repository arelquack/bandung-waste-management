# --- Library Eksternal ---
import fitz  # PyMuPDF untuk ekstraksi teks dari PDF
import re  # Regex untuk pembersihan teks
import joblib  # Untuk serialisasi model atau data
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF untuk indexing teks
import nltk  # Natural Language Toolkit untuk tokenisasi
from nltk.tokenize import word_tokenize
import os
import argparse  # Untuk parsing argumen CLI
import logging  # Logging untuk pelacakan proses
import unittest  # Unit testing untuk validasi fungsi

# --- Konfigurasi Logging Default ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path):
    """
    Mengekstrak teks dari file PDF menggunakan PyMuPDF (fitz).
    
    Args:
        pdf_path (str): Path ke file PDF.
    
    Returns:
        str | None: Seluruh teks dari PDF, atau None jika gagal.
    """
    try:
        if not os.path.exists(pdf_path):
            logging.error(f"File tidak ditemukan di: {pdf_path}")
            return None
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text
        doc.close()
        if not text.strip():
            logging.warning("Dokumen PDF tampaknya kosong atau tidak berisi teks yang dapat diekstrak.")
            return None
        return text
    except Exception as e:
        logging.error(f"Error saat memproses PDF: {e}")
        return None

def preprocess_text(text):
    """
    Membersihkan teks mentah dari hasil ekstraksi PDF.
    - Menghapus header/footer
    - Menghapus referensi hukum yang berulang
    - Merapikan whitespace
    
    Args:
        text (str): Teks mentah.
    
    Returns:
        str: Teks yang telah dibersihkan.
    """
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'Peraturan Daerah Nomor 9 Tahun 2018', '', text, flags=re.IGNORECASE)
    
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def chunk_text(text, chunk_size=300, overlap=50):
    """
    Membagi teks panjang menjadi beberapa potongan (chunk) dengan overlap token.
    
    Args:
        text (str): Teks yang ingin dipecah.
        chunk_size (int): Jumlah maksimal token per chunk.
        overlap (int): Jumlah token yang tumpang tindih antar chunk.
    
    Returns:
        List[str]: List berisi potongan teks (chunk).
    """
    try:
        tokens = word_tokenize(text)
    except LookupError:
        logging.warning("NLTK 'punkt' resource not found. Downloading...")
        nltk.download('punkt')
        tokens = word_tokenize(text)
    
    chunks = []
    
    if chunk_size <= overlap:
        logging.error("chunk_size harus lebih besar dari overlap.")
        return []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = ' '.join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_tfidf_index(chunks):
    """
    Membuat indeks TF-IDF dari daftar chunk teks.
    
    Args:
        chunks (List[str]): Potongan teks.
    
    Returns:
        Tuple[vectorizer, matrix]: TF-IDF vectorizer dan matriksnya.
    """
    if not chunks:
        logging.warning("Tidak ada chunks untuk diindeks.")
        return None, None
    import numpy as np
    # gunakan konfigurasi yang lebih efisien dan akurat untuk definisi
    vectorizer = TfidfVectorizer(dtype=np.float32, sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def analyze_chunks(chunks):
    """
    Menganalisis statistik dasar dari chunks yang dibuat.
    
    Args:
        chunks (List[str]): Potongan teks hasil pemrosesan.
    """
    if not chunks:
        logging.info("Tidak ada chunks untuk dianalisis.")
        return
    
    chunk_lengths = [len(word_tokenize(chunk)) for chunk in chunks]
    if not chunk_lengths:
        logging.info("Tidak ada chunks untuk dianalisis.")
        return

    avg_length = sum(chunk_lengths) / len(chunks)
    
    logging.info("\n--- Analisis Chunks ---")
    logging.info(f"Total chunks: {len(chunks)}")
    logging.info(f"Rata-rata token per chunk: {avg_length:.2f}")
    logging.info("\n3 Sample Chunk Pertama:")
    for i, chunk in enumerate(chunks[:3]):
        logging.info(f"Chunk {i+1} (panjang {len(word_tokenize(chunk))} token):\n{chunk[:200]}...")

class TestPreprocessing(unittest.TestCase):
    """
    Unit test untuk fungsi preprocessing dan chunking.
    """
    def test_preprocess_text(self):
        text = "Peraturan Daerah Nomor 9 Tahun 2018\n\n Ini adalah teks.   \n\n Page 1 of 1"
        expected = "Ini adalah teks."
        self.assertEqual(preprocess_text(text), expected)

    def test_chunk_text(self):
        long_text = "Ini adalah sebuah contoh teks yang panjang untuk dipecah menjadi beberapa chunk. Kita akan memastikan bahwa proses chunking berjalan dengan benar dan ada tumpang tindih antar chunk."
        chunks = chunk_text(long_text, chunk_size=20, overlap=5)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0], 'Ini adalah sebuah contoh teks yang panjang untuk dipecah menjadi beberapa chunk .')

    @unittest.skip("Membutuhkan file dummy untuk testing.")
    def test_extract_text_from_pdf(self):
        pass

def run_tests():
    """
    Menjalankan semua unit test yang telah didefinisikan.
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTest(loader.loadTestsFromTestCase(TestPreprocessing))
    runner = unittest.TextTestRunner()
    runner.run(suite)

def main():
    """
    Fungsi utama untuk menjalankan pipeline pemrosesan PDF menjadi TF-IDF pickle.
    """
    parser = argparse.ArgumentParser(description='Script untuk memproses dokumen PERDA dan membuat TF-IDF index.')
    parser.add_argument('pdf_path', nargs='?', type=str, help='Path ke file PDF PERDA.')
    parser.add_argument('--output', type=str, default="perda_data.pkl", help='Lokasi file output pickle.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Atur level logging.')
    parser.add_argument('--run-tests', action='store_true', help='Jalankan unit tests.')
    args = parser.parse_args()
    
    # Atur level logging sesuai dengan parameter
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    if args.run_tests:
        run_tests()
        return

    if not args.pdf_path:
        logging.error("Argumen 'pdf_path' wajib disertakan. Contoh: python perda_processor.py <path_ke_pdf>")
        return

    logging.info("Memulai proses persiapan data...")
    
    # 1. Ekstrak teks dari PDF
    raw_text = extract_text_from_pdf(args.pdf_path)
    if raw_text is None:
        return
    
    # 2. Praproses teks
    cleaned_text = preprocess_text(raw_text)
    
    # 3. Potong teks menjadi chunks
    document_chunks = chunk_text(cleaned_text, chunk_size=300, overlap=50)
    if not document_chunks:
        logging.error("Gagal membagi dokumen menjadi chunks.")
        return
    
    # 4. Analisis statistik chunks
    analyze_chunks(document_chunks)
    
    # 5. Buat indeks TF-IDF
    vectorizer, tfidf_matrix = create_tfidf_index(document_chunks)
    if vectorizer is None or tfidf_matrix is None:
        logging.error("Gagal membuat TF-IDF index.")
        return
    # 6. Simpan hasil ke dalam file terpisah (vectorizer, matrix, chunks)
    try:
        from scipy.sparse import save_npz
        base, _ = os.path.splitext(args.output)
        vec_path = base + "_vectorizer.pkl"
        mat_path = base + "_tfidf.npz"
        chunks_path = base + "_chunks.pkl"
        # simpan artefak
        joblib.dump(vectorizer, vec_path, compress=3)
        save_npz(mat_path, tfidf_matrix)
        joblib.dump(document_chunks, chunks_path, compress=3)
        logging.info(f"Artefak disimpan: {vec_path}, {mat_path}, {chunks_path}")
    except Exception as e:
        logging.error(f"Gagal menyimpan artefak terpisah: {e}")
        return
    # tetap simpan gabungan untuk kompatibilitas mundur
    processed_data = {
        'chunks': document_chunks,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix
    }
    joblib.dump(processed_data, args.output)
    logging.info(f"\nProses selesai. Data berhasil disimpan ke {args.output}")
    logging.info(f"Ukuran TF-IDF matrix: {tfidf_matrix.shape}")

if __name__ == "__main__":
    main()