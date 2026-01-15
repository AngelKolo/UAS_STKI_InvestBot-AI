# -*- coding: utf-8 -*-
"""
InvestBot AI - Streamlit Version
UAS SISTEM TEMU KEMBALI INFORMASI (STKI) GANJIL 2025/2026

Nama   : Angelica Widyastuti Kolo
NIM    : A11.2021.13212
Kelas  : A11.4706
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import nltk
from typing import List
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ================================================================================
# CONFIGURATION
# ================================================================================
st.set_page_config(
    page_title="InvestBot AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2B6CB0;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2B6CB0;
        margin: 1rem 0;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #ffc107;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# DOWNLOAD NLTK DATA (ONE-TIME)
# ================================================================================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# ================================================================================
# TEXT PREPROCESSING
# ================================================================================
class TextPreprocessor:
    def __init__(self):
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.stopwords = set(['yang', 'di', 'ke', 'dari', 'dan', 'ini', 'itu', 
                              'adalah', 'ada', 'untuk', 'pada', 'dengan', 'atau'])

    def preprocess(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = text.split()
        clean_tokens = [self.stemmer.stem(t) for t in tokens 
                       if t not in self.stopwords and len(t) > 2]
        return clean_tokens

# ================================================================================
# TF-IDF IMPLEMENTATION
# ================================================================================
class TFIDFFromScratch:
    def __init__(self):
        self.vocabulary = {}
        self.idf = {}
        self.n_docs = 0

    def fit(self, docs_tokens: List[List[str]]):
        self.n_docs = len(docs_tokens)
        df_counts = Counter()
        all_terms = set()
        
        for doc in docs_tokens:
            unique_terms = set(doc)
            all_terms.update(unique_terms)
            for term in unique_terms:
                df_counts[term] += 1

        self.vocabulary = {term: i for i, term in enumerate(sorted(list(all_terms)))}
        for term, df in df_counts.items():
            self.idf[term] = math.log(self.n_docs / (1 + df))

    def transform(self, docs_tokens: List[List[str]]) -> np.ndarray:
        matrix = np.zeros((len(docs_tokens), len(self.vocabulary)))
        for i, doc in enumerate(docs_tokens):
            tf_counts = Counter(doc)
            for term, count in tf_counts.items():
                if term in self.vocabulary:
                    tf = count / len(doc) if len(doc) > 0 else 0
                    matrix[i, self.vocabulary[term]] = tf * self.idf[term]
        return matrix

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if norm_v1 > 0 and norm_v2 > 0 else 0

# ================================================================================
# SENTIMENT ANALYSIS - K-NN CLASSIFIER
# ================================================================================
class KNNSentimentClassifier:
    """K-NN Classifier untuk sentiment analysis menggunakan dataset"""
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.preprocessor = None
        self.tfidf = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)

    def predict(self, x_query):
        similarities = [cosine_similarity(x_query, x_t) for x_t in self.X_train]
        k_indices = np.argsort(similarities)[-self.k:][::-1]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

class SentimentAnalyzer:
    """Hybrid: ML-based (K-NN) + Lexicon fallback"""
    def __init__(self, use_ml=True):
        self.use_ml = use_ml
        self.knn_model = None
        self.sentiment_tfidf = None
        self.sentiment_preprocessor = None
        
        # Lexicon fallback
        self.pos_words = set(['untung', 'naik', 'bagus', 'cuan', 'aman', 'stabil', 
                             'rekomendasi', 'tumbuh', 'profit', 'menguntung', 
                             'positif', 'baik', 'sukses', 'mantap', 'joss'])
        self.neg_words = set(['rugi', 'turun', 'anjlok', 'bahaya', 'resiko', 'buruk', 
                             'waspada', 'bangkrut', 'loss', 'negatif', 'jelek', 'gagal'])
    
    def train_from_dataset(self, df_sentiment):
        """Train K-NN model dari sentiment_opinions.csv"""
        try:
            self.sentiment_preprocessor = TextPreprocessor()
            self.sentiment_tfidf = TFIDFFromScratch()
            
            # Preprocess sentiment data
            processed_sents = [self.sentiment_preprocessor.preprocess(str(text)) 
                              for text in df_sentiment['text']]
            
            # Fit TF-IDF
            self.sentiment_tfidf.fit(processed_sents)
            X_sentiment = self.sentiment_tfidf.transform(processed_sents)
            
            # Train K-NN
            self.knn_model = KNNSentimentClassifier(k=5)
            self.knn_model.fit(X_sentiment, df_sentiment['label'].values)
            
            return True
        except Exception as e:
            print(f"Warning: Gagal train sentiment model: {e}")
            return False
    
    def analyze(self, text):
        """Analyze sentiment with ML or fallback to lexicon"""
        # Try ML-based first
        if self.use_ml and self.knn_model is not None:
            try:
                tokens = self.sentiment_preprocessor.preprocess(text)
                vec = self.sentiment_tfidf.transform([tokens])[0]
                prediction = self.knn_model.predict(vec)
                
                # Map label ke format output
                label_map = {
                    'positif': ('Positif', 'sentiment-positive'),
                    'negatif': ('Negatif', 'sentiment-negative'),
                    'netral': ('Netral', 'sentiment-neutral')
                }
                return label_map.get(prediction.lower(), ('Netral', 'sentiment-neutral'))
            except:
                pass
        
        # Fallback to lexicon-based
        tokens = text.lower().split()
        score = (sum(1 for t in tokens if t in self.pos_words) - 
                sum(1 for t in tokens if t in self.neg_words))
        
        if score > 0:
            return "Positif", "sentiment-positive"
        elif score < 0:
            return "Negatif", "sentiment-negative"
        else:
            return "Netral", "sentiment-neutral"

# ================================================================================
# EXTRACTIVE SUMMARIZATION
# ================================================================================
class ExtractiveSummarizer:
    def summarize(self, text, n_sentences=3):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= n_sentences:
            return text

        words = text.lower().split()
        word_freq = Counter(words)

        sent_scores = []
        for sent in sentences:
            score = sum(word_freq.get(w.lower(), 0) for w in sent.split())
            sent_scores.append(score)

        top_indices = np.argsort(sent_scores)[-n_sentences:]
        return " ".join([sentences[i] for i in sorted(top_indices)])

# ================================================================================
# SMART ANSWER GENERATOR (QUERY-AWARE)
# ================================================================================
class SmartAnswerGenerator:
    """Generate query-aware answers using intelligent sentence selection"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def generate_answer(self, context: str, query: str, category: str) -> str:
        # Split context into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        if len(sentences) == 0:
            return "Maaf, tidak dapat memproses konteks dokumen."
        
        # Preprocess query untuk mendapatkan keywords
        query_tokens = set(self.preprocessor.preprocess(query))
        
        # Score setiap kalimat berdasarkan relevansi dengan query
        scored_sentences = []
        for sent in sentences:
            sent_tokens = set(self.preprocessor.preprocess(sent))
            # Hitung overlap antara query dan sentence (Jaccard similarity)
            overlap = len(query_tokens & sent_tokens)
            jaccard = overlap / len(query_tokens | sent_tokens) if len(query_tokens | sent_tokens) > 0 else 0
            scored_sentences.append((sent, jaccard, overlap))
        
        # Sort by relevance score (descending)
        scored_sentences.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Ambil top 3-5 kalimat paling relevan
        n_sentences = min(4, len(scored_sentences))
        top_sentences = [s[0] for s in scored_sentences[:n_sentences] if s[1] > 0]
        
        # Jika tidak ada yang relevan, ambil awal dokumen
        if not top_sentences:
            top_sentences = sentences[:3]
        
        # Gabungkan dengan natural flow
        answer = " ".join(top_sentences)
        
        # Cleanup: hilangkan kalimat yang terlalu pendek (<20 char)
        final_sentences = [s for s in answer.split('. ') if len(s.strip()) > 20]
        answer = ". ".join(final_sentences)
        
        # Pastikan ada titik di akhir
        if answer and not answer.endswith('.'):
            answer += '.'
        
        return answer if answer else context[:500] + "..."

# ================================================================================
# LOAD DATA & INITIALIZE MODELS
# ================================================================================
@st.cache_resource
def load_system():
    """Load all components once and cache them"""
    try:
        # Load main dataset
        df = pd.read_csv('classified_documents.csv')
        
        # Initialize components
        preprocessor = TextPreprocessor()
        tfidf = TFIDFFromScratch()
        
        # Preprocess documents
        processed_docs = [preprocessor.preprocess(str(text)) for text in df['text']]
        tfidf.fit(processed_docs)
        X_matrix = tfidf.transform(processed_docs)
        
        # Initialize sentiment analyzer
        sentiment_tool = SentimentAnalyzer(use_ml=True)
        
        # Try to load and train sentiment model
        try:
            df_sentiment = pd.read_csv('sentiment_opinions.csv')
            if sentiment_tool.train_from_dataset(df_sentiment):
                sentiment_status = f"✅ K-NN Sentiment (trained on {len(df_sentiment)} samples)"
            else:
                sentiment_status = "⚠️ Lexicon-based Sentiment (fallback)"
        except FileNotFoundError:
            sentiment_status = "⚠️ Lexicon-based Sentiment (sentiment_opinions.csv not found)"
        except Exception as e:
            sentiment_status = f"⚠️ Lexicon-based Sentiment (error: {str(e)[:50]})"
        
        # Initialize answer generator
        answer_generator = RuleBasedAnswerGenerator()
        
        return df, preprocessor, tfidf, X_matrix, sentiment_tool, answer_generator, None, sentiment_status
    
    except Exception as e:
        st.error(f"Error loading system: {str(e)}")
        return None, None, None, None, None, None, str(e), "Error"

# ================================================================================
# MAIN CHATBOT FUNCTION
# ================================================================================
def invest_bot_response(query, df, preprocessor, tfidf, X_matrix, sentiment_tool, answer_generator):
    """Generate response for user query"""
    
    if not query.strip():
        return "Silakan masukkan pertanyaan Anda.", "Netral", "sentiment-neutral", "-", "-"
    
    try:
        # 1. Preprocess query
        q_tokens = preprocessor.preprocess(query)
        if not q_tokens:
            return "Maaf, tidak dapat memproses pertanyaan. Coba gunakan kata kunci yang lebih spesifik.", "Netral", "sentiment-neutral", "-", "-"
        
        q_vec = tfidf.transform([q_tokens])[0]
        
        # 2. Retrieval (Find most similar document)
        sims = [cosine_similarity(q_vec, d_vec) for d_vec in X_matrix]
        top_idx = np.argmax(sims)
        similarity_score = sims[top_idx]
        
        # Check if similarity too low
        if similarity_score < 0.1:
            return ("Maaf, saya tidak menemukan informasi yang cukup relevan dalam database. "
                   "Coba tanyakan tentang saham, crypto, emas, reksadana, atau properti."), "Netral", "sentiment-neutral", "-", "0.0"
        
        context = str(df.iloc[top_idx]['text'])
        source = str(df.iloc[top_idx]['category'])
        doc_id = str(df.iloc[top_idx]['doc_id'])
        
        # 3. Sentiment Analysis
        sentiment, sentiment_class = sentiment_tool.analyze(query)
        
        # 4. Generate Answer (Rule-based, no LLM)
        answer = answer_generator.generate_answer(context, query, source)
        
        return answer, sentiment, sentiment_class, f"{source} (Doc ID: {doc_id})", f"{similarity_score:.2f}"
    
    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}", "Netral", "sentiment-neutral", "-", "-"

# ================================================================================
# STREAMLIT UI
# ================================================================================
def main():
    # Header
    st.markdown('<div class="main-header">💰 InvestBot AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Asisten Cerdas Edukasi Investasi - RAG System dengan Sentiment Analysis</div>', unsafe_allow_html=True)
    
    # Load system
    with st.spinner('🔄 Memuat sistem... Mohon tunggu'):
        df, preprocessor, tfidf, X_matrix, sentiment_tool, answer_generator, error, sentiment_status = load_system()
    
    if error:
        st.error(f"❌ Gagal memuat sistem: {error}")
        st.info("💡 Pastikan file 'classified_documents.csv' ada di direktori yang sama dengan app.py")
        return
    
    st.success(f"✅ Sistem siap! • {sentiment_status}")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Informasi")
        st.markdown(f"""
        **Dataset:** {len(df)} dokumen  
        **Kategori:**
        - 📈 Saham
        - 💎 Cryptocurrency
        - 🥇 Emas
        - 📊 Reksa Dana
        - 🏠 Properti
        
        **Fitur:**
        - ✅ RAG System (TF-IDF)
        - ✅ K-NN Sentiment Classifier
        - ✅ Smart Retrieval
        - ✅ Auto Summarization
        
        **Sentiment Model:**
        {sentiment_status}
        """)
        
        st.markdown("---")
        st.markdown("**Contoh Pertanyaan:**")
        
        examples = [
            "Apa itu investasi saham?",
            "Bagaimana cara investasi crypto?",
            "Apa keuntungan investasi emas?",
            "Tips diversifikasi portofolio",
            "Risiko investasi properti"
        ]
        
        for ex in examples:
            if st.button(ex, key=ex):
                st.session_state.query_input = ex
        
        st.markdown("---")
        st.caption("**Mahasiswa:**")
        st.caption("Angelica Widyastuti Kolo")
        st.caption("A11.2021.13212")
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Pertanyaan Anda")
        query = st.text_area(
            "Masukkan pertanyaan tentang investasi:",
            height=150,
            placeholder="Contoh: Bagaimana cara investasi saham untuk pemula?",
            key="query_input"
        )
        
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            submit = st.button("🚀 Kirim Pertanyaan", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Hapus", use_container_width=True):
                st.session_state.query_input = ""
                st.rerun()
    
    with col2:
        st.subheader("💡 Jawaban")
        
        if submit and query:
            with st.spinner('🤔 Sedang menganalisis pertanyaan Anda...'):
                answer, sentiment, sent_class, source, similarity = invest_bot_response(
                    query, df, preprocessor, tfidf, X_matrix, sentiment_tool, answer_generator
                )
            
            # Display answer
            st.markdown(f'<div class="result-box">{answer}</div>', unsafe_allow_html=True)
            
            # Display metadata
            st.markdown("---")
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            
            with col_meta1:
                st.markdown(f"**📊 Sentimen:**")
                st.markdown(f'<span class="{sent_class}">{sentiment}</span>', unsafe_allow_html=True)
            
            with col_meta2:
                st.markdown(f"**📚 Sumber:**")
                st.text(source)
            
            with col_meta3:
                st.markdown(f"**🎯 Relevansi:**")
                st.text(similarity)
        
        elif submit and not query:
            st.warning("⚠️ Silakan masukkan pertanyaan terlebih dahulu.")
        else:
            st.info("👆 Masukkan pertanyaan dan klik tombol Kirim untuk mendapatkan jawaban.")
    
    # Footer
    st.markdown('<div class="footer">UAS Sistem Temu Kembali Informasi 2025/2026 • Universitas Dian Nuswantoro</div>', 
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
