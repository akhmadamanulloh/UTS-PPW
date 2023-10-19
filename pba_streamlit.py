import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans  
from sklearn import tree

Data, Ekstraksi, lda, LDAkmeans, Model ,implementasi = st.tabs(['Data', 'Ekstraksi Fitur', 'LDA', 'LDA kmeans', 'Modelling', 'implementasi'])

with Data :
   st.title("""UTS PPW A""")
   st.text('Akhmad Amanulloh 200411100099')
   st.subheader('Data')
   data=pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/crawling_pta_labeled.csv')
   data

with Ekstraksi :

   st.subheader('Term Frequency (TF)')
   tf = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/TF.csv')
   tf
   
   st.subheader('Logarithm Frequency (Log-TF)')
   log_tf = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/log_TF.csv')
   log_tf
   
   st.subheader('One Hot Encoder / Binary')
   oht = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/OneHotEncoder.csv')
   oht
   
   st.subheader('TF-IDF')
   tf_idf = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/TF-IDF.csv')
   tf_idf

with lda:
        lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
        x=tf.drop('Label', axis=1)
        lda_top=lda.fit_transform(x)
        U = pd.DataFrame(lda_top, columns=['Topik 1','Topik 2','Topik 3'])
        U['Label']=tf['Label'].values
        U

with LDAkmeans:
      kmeans = KMeans(n_clusters=3, random_state=0)
      x=tf.drop('Label', axis=1)
      clusters = kmeans.fit_predict(x)   
      U['Cluster'] = clusters
      U['Label']=tf['Label'].values
      U
   
with Model :
    # if all :
        lda = LatentDirichletAllocation(n_components=3, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
        x=tf.drop('Label', axis=1)
        lda_top=lda.fit_transform(x)
        y = tf.Label
        X_train,X_test,y_train,y_test = train_test_split(lda_top,y,test_size=0.2,random_state=42)
        
        metode1 = KNeighborsClassifier(n_neighbors=3)
        metode1.fit(X_train, y_train)

        metode2 = GaussianNB()
        metode2.fit(X_train, y_train)

        metode3 = tree.DecisionTreeClassifier(criterion="gini")
        metode3.fit(X_train, y_train)

        st.write ("Pilih metode yang ingin anda gunakan :")
        met1 = st.checkbox("KNN")
        met2 = st.checkbox("Naive Bayes")
        met3 = st.checkbox("Decesion Tree")
        submit2 = st.button("Pilih")

        if submit2:      
            if met1 :
                st.write("KNN")
                st.write("Hasil Akurasi Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
            elif met2 :
                st.write("Naive Bayes")
                st.write("Hasil Akurasi Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
            elif met3 :
                st.write("Decesion Tree")
                st.write("Hasil Akurasi Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
            else :
                st.write("Anda Belum Memilih Metode")
with implementasi:
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        # Baca data
        data_x = pd.read_csv('https://raw.githubusercontent.com/akhmadamanulloh/pba/main/Term%20Frequensi%20Berlabel%20Final.csv')
        data_x = data_x.dropna(subset=['Dokumen'])  # Menghapus baris yang memiliki NaN di kolom 'Dokumen'

        # Ubah kelas A menjadi 0 dan kelas B menjadi 1
        kelas_dataset_binary = [0 if kelas == 'RPL' else 1 for kelas in data_x['Label']]
        data_x['Label'] = kelas_dataset_binary

        # Bagi data menjadi data pelatihan dan data pengujian
        X = data_x['Dokumen']
        label = data_x['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

        # Vektorisasi teks menggunakan TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Latih model Naive Bayes
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train_tfidf, y_train)

        # Latih model LDA
        k = 3
        alpha = 0.1
        beta = 0.2
        lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
        proporsi_topik_dokumen = lda.fit_transform(X_train_tfidf)

        

        with st.form("my_form"):
            st.subheader("Implementasi")
            input_dokumen = st.text_input('Masukkan Judul Yang Akan Diklasfifikasi')
            input_vector = tfidf_vectorizer.transform([input_dokumen])
            submit = st.form_submit_button("submit")
            # Prediksi proporsi topik menggunakan model LDA
            proporsi_topik = lda.transform(input_vector)[0]
            if submit:
                st.subheader('Hasil Prediksi')
                inputs = np.array([input_dokumen])
                input_norm = np.array(inputs)
                input_pred = nb_classifier.predict(input_vector)[0]
            # Menampilkan hasil prediksi
                if input_pred==0:
                    st.success('RPL')
                    st.write(proporsi_topik)
                else  :
                    st.success('KK')
                    st.write(proporsi_topik)
