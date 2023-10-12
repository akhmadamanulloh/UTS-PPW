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

Data, Ekstraksi, lda, LDAkmeans, Model = st.tabs(['Data', 'Ekstraksi Fitur', 'LDA', 'LDA kmeans', 'Modelling'])

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
        # if met1 :
        #     st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
        #     st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
        met2 = st.checkbox("Naive Bayes")
        # if met2 :
        #     st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
        #     st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
        met3 = st.checkbox("Decesion Tree")
        # if met3 :
            # st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
            # st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
        submit2 = st.button("Pilih")

        if submit2:      
            if met1 :
                st.write("Metode yang Anda gunakan Adalah KNN")
                st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
            elif met2 :
                st.write("Metode yang Anda gunakan Adalah Naive Bayes")
                st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
            elif met3 :
                st.write("Metode yang Anda gunakan Adalah Decesion Tree")
                st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
            else :
                st.write("Anda Belum Memilih Metode")
    # else:
    #     st.write("Anda Belum Menentukan Jumlah Topik di Menu LDA")
