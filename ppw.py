import warnings
import pandas as pd
import numpy as np
import nltk
import streamlit as st 

import re 
import csv

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.decomposition import LatentDirichletAllocation 

nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings('ignore')

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Hasil Dataset", "Hasil Cleaning Data", 'Hasil Tokenizing Data', 'Hasil Stopword Data', 'Hasil TF-IDF Data', 'Dataset dari TermFrequensi', 'Hasil Topik pada Dokumen', 'Hasil Kata pada Topik'])


with tab1 :
    csv_path = 'https://raw.githubusercontent.com/arshell19/DATASET/main/dataPTATrunojoyo%20(3).csv'
    df = pd.read_csv(csv_path)
    df

with tab2 :
    def cleaning(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text).strip()
        return text
    
    df['data_clean'] = df['Abstrak'].apply(cleaning)
    df['data_clean']

with tab3 :
    def tokenizer(text):
        text = text.lower()
        return word_tokenize(text)

    df['Tokenizing'] = df['data_clean'].apply(tokenizer)
    df['Tokenizing']

with tab4 :
    corpus = stopwords.words('indonesian')

    def stopwordText(words):
        return [word for word in words if word not in corpus]

    df['Stopword Removal'] = df['Tokenizing'].apply(stopwordText)

    # Gabungkan kembali token menjadi kalimat utuh
    df['stopword'] = df['Stopword Removal'].apply(lambda x: ' '.join(x))
    df['stopword']

with tab5 :
    def tfidf(dokumen):
        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform(dokumen).toarray()
        terms = vectorizer.get_feature_names_out()

        final_tfidf = pd.DataFrame(x, columns=terms)
        final_tfidf.insert(0, 'Abstrak', dokumen)

        return (vectorizer, final_tfidf)

    tfidf_vectorizer, final_tfidf = tfidf(df['stopword'])
    final_tfidf 

with tab6 :
    csv_path = 'https://raw.githubusercontent.com/arshell19/DATASET/main/HasilTermFrequensi-new.csv'
    df = pd.read_csv(csv_path)
    df

with tab7 :
    X = df.drop('Abstrak', axis=1)
    k = 2
    alpha = 0.1
    beta = 0.2

    lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    proporsi_topik_dokumen = lda.fit_transform(X)
    #proporsi topik pada dokumen
    dokumen = df['Abstrak']
    output_proporsi_TD = pd.DataFrame(proporsi_topik_dokumen, columns=['Topik_1', 'Topik_2'])
    output_proporsi_TD.insert(0,'Abstrak', dokumen)
    output_proporsi_TD

with tab8 :
    #proporsi kata pada topik
    distribusi_kata_topik = pd.DataFrame(lda.components_)
    distribusi_kata_topik
