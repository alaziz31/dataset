import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import os
import warnings
import altair as alt
from sklearn.utils.validation import joblib

st.set_page_config(page_title="Body Performance", page_icon='icon.png', layout="wide", initial_sidebar_state="auto")

st.title("UAS PENAMBANGAN DATA")

description, importdata, implementation = st.tabs(["Deskripsi ", " Import Data ", " Implementation"])
# warnings.filterwarnings("ignore")
with description:
    st.subheader("Deskripsi")
    st.write("Nama : CITRA INDAH LESTARI | NIM : 200411100202 | Kelas : Penambangan Data A")
    st.write("")
    st.write("Dataset berisi tentang prediksi lulus atau gagalsesuai dengan nilai siswa.")
    st.write("Aplikasi ini digunakan untuk memprediksi kelulusan siswa lewat nilai.")
    st.write("Fitur yang digunakan :")
    st.write("1. school (Sekolah : Numerik")
    st.write("2. sex (Gender : numerik")
    st.write("3. age (Usia : Numerik")
    st.write("4. address (Alamat : Numerik")
    st.write("5. famsize (ukuran keluarga : Numerik")
    st.write("6. Pstatus (status kohabitasi orang tua : Numerik")
    st.write("7. medu (pendidikan ibu : Numerik")
    st.write("8. Fedu  (pendidikan ayah : Numerik")
    st.write("9. traveltime (waktu tempuh dari rumah ke sekolah : Numerik")
    st.write("10. studytime (waktu belajar mingguan : Numerik")
    st.write("11. failures (jumlah kegagalan kelas sebelumnya : Numerik")
    st.write("12. schoolsup (dukungan pendidikan ekstra : Numerik")
    st.write("13. famsup (dukungan pendidikan keluarga : Numerik")
    st.write("14. paid (kelas berbayar ekstra dalam mata pelajaran kursus : Numerik")
    st.write("15. activities (kegiatan ekstrakurikuler : Numerik")
    st.write("16. nursery (menghadiri sekolah pembibitan : Numerik")
    st.write("17. higher (ingin mengambil pendidikan tinggi : Numerik")
    st.write("18. internet (Akses internet di rumah : Numerik")
    st.write("19. romantic (dengan hubungan romantis : Numerik")
    st.write("20. famrel (kualitas hubungan keluarga : Numerik")
    st.write("21. freetime (waktu luang setelah sekolah : Numerik")
    st.write("22. goout (pacaran dengan teman : Numerik")
    st.write("23. Dalc (konsumsi alkohol pada hari kerja : Numerik")
    st.write("24. Walc (konsumsi alkohol akhir pekan : Numerik")
    st.write("25. health (status kesehatan saat ini : Numerik")
    st.write("26. absences (jumlah absen sekolah : Numerik")
    st.write("27. G1 (nilai periode pertama : Numerik")
    st.write("28. G2 (nilai periode kedua : Numerik")
    st.write("29. G3 (nilai akhir : Numerik")
    st.write("30. pass (dukungan pendidikan ekstra : Numerik")
    st.write("Sumber dataset https://www.kaggle.com/datasets/dinhanhx/studentgradepassorfailprediction")
    st.write("Link github https://github.com/CitraIndahL/dataset")

with importdata:
    for uploaded_file in uploaded_files:
        dataset, preprocessing, modelling = st.tabs(["Dataset", "Preprocessing", "Modelling"])
        with dataset:
            st.write("Import Data")
            data = pd.read_csv("https://raw.githubusercontent.com/CitraIndahL/dataset/main/student-mat-pass-or-fail.csv")
            st.dataframe(data)
        with preprocessing:
            st.subheader("Preprocessing")
            prepros = st.radio(
            "Silahkan pilih metode yang digunakan :",
            (["Min Max Scaler"]))
            prepoc = st.button("Preprocessing")

            if prepros == "Min Max Scaler":
                if prepoc:
                    df[["school","sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "traveltime", "studytime","failures","schoolsup","famsup", "paid", "activities", "nursery", "higher", "internet", "romantic","famrel","freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3", "pass"]].agg(['min','max'])
                    df.pass.value_counts()
                    X = df.drop(columns=["pass"],axis=1)
                    y = df["pass"]

                    "### Normalize data transformasi"
                    X
                    X.shape, y.shape
                    # le.inverse_transform(y)
                    labels = pd.get_dummies(df.Class).columns.values.tolist()
                    "### Label"
                    labels
                    """## Normalisasi MinMax Scaler"""
                    scaler = MinMaxScaler()
                    scaler.fit(X)
                    X = scaler.transform(X)
                    X
                    X.shape, y.shape

        with modelling:
            X=df[["school","sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "traveltime", "studytime","failures","schoolsup","famsup", "paid", "activities", "nursery", "higher", "internet", "romantic","famrel","freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2", "G3", "pass"]]
            y=df["pass"]
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            # from sklearn.feature_extraction.text import CountVectorizer
            # cv = CountVectorizer()
            # X_train = cv.fit_transform(X_train)
            # X_test = cv.fit_transform(X_test)
            st.subheader("Modeling")
            st.write("Silahkan pilih Model :")
            naive = st.checkbox('Naive Bayes')
            kn = st.checkbox('K-Nearest Neighbor')
            des = st.checkbox('Decision Tree')
            mod = st.button("Modeling")

            # NB
            GaussianNB(priors=None)

            # Fitting Naive Bayes Classification to the Training set with linear kernel
            nvklasifikasi = GaussianNB()
            nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

            # Predicting the Test set results
            y_pred = nvklasifikasi.predict(X_test)
            
            y_compare = np.vstack((y_test,y_pred)).T
            nvklasifikasi.predict_proba(X_test)
            akurasi_nb = round(100 * accuracy_score(y_test, y_pred))
            # akurasi_nb = 10

            # KNN 
            K=10
            knn=KNeighborsClassifier(n_neighbors=K)
            knn.fit(X_train,y_train)
            y_pred=knn.predict(X_test)

            akurasi_knn = round(100 * accuracy_score(y_test,y_pred))

            # DT

            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)
            # prediction
            dt.score(X_test, y_test)
            y_pred = dt.predict(X_test)
            #Accuracy
            akurasi_dt = round(100 * accuracy_score(y_test,y_pred))

            if naive :
                if mod :
                    st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi_nb))
            if kn :
                if mod:
                    st.write("Model KNN accuracy score : {0:0.2f}" . format(akurasi_knn))
            if des :
                if mod :
                    st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasi_dt))


    
with implementation:
    st.subheader("Implementation")
    school = st.number_input('Masukkan sekolah siswa (1 : Garbiel Pereira, 0 : Mousinho da Silveria )')
    sex = st.number_input('Masukkan gender ( 1=Pria 0=Wanita )')
    age = st.number_input('Masukkan Usia (Tahun)')
    address = st.number_input('Masukkan Berat badan (Kg)')
    famsize = st.number_input('Masukkan Lemak tubuh (Persen)')
    Pstatus = st.number_input('Masukkan Tekanan diastolic')
    Medu = st.number_input('Masukkan Tekanan systolic')
    Fedu = st.number_input('Masukkan Kekuatan cengkraman (Kg)')
    traveltime = st.number_input('Masukkan Ukuran sit and bend (cm)')
    studytime = st.number_input('Masukkan Jumlah sit-ups')
    failures = st.number_input('Masukkan Jarak broad jump (cm)')
    schoolsup = st.number_input('Masukkan Usia (tahun)')
    famsup = st.number_input('Masukkan gender ( 1=Pria 0=Wanita )')
    paid = st.number_input('Masukkan Tinggi badan (cm)')
    activities = st.number_input('Masukkan Berat badan (Kg)')
    nursery = st.number_input('Masukkan Lemak tubuh (Persen)')
    higher = st.number_input('Masukkan Tekanan diastolic')
    internet = st.number_input('Masukkan Tekanan systolic')
    romantic = st.number_input('Masukkan Kekuatan cengkraman (Kg)')
    famrel = st.number_input('Masukkan Ukuran sit and bend (cm)')
    freetime = st.number_input('Masukkan Jumlah sit-ups')
    goout = st.number_input('Masukkan Jarak broad jump (cm)')
    Dalc = st.number_input('Masukkan Jarak broad jump (cm)')
    Walc = st.number_input('Masukkan Usia (tahun)')
    health = st.number_input('Masukkan gender ( 1=Pria 0=Wanita )')
    absences = st.number_input('Masukkan Tinggi badan (cm)')
    G1 = st.number_input('Masukkan Berat badan (Kg)')
    G2 = st.number_input('Masukkan Lemak tubuh (Persen)')
    G3 = st.number_input('Masukkan Tekanan diastolic')
    pass = st.number_input('Masukkan Tekanan systolic')

    def submit():
        # input
        inputs = np.array([[
            school, sex, age, address, famsize, Pstatus, Medu, Fedu, traveltime, studytime, failures,schoolsup, famsup, paid, activities, nursery, higher, internet, romantic, famrel, freetime, goout, Dalc, Walc, health, absences, G1, G2, G3, pass
        ]])
        baru = pd.DataFrame(inputs)
        input = pd.get_dummies(baru)
        st.write("Data yang diinputkan :")
        st.write(input)
        inputan = np.array(input)
        le = joblib.load("le.save")
        model1 = joblib.load("tree.joblib")
        y_pred3 = model1.predict(inputs)
        st.write("Berdasarkan data yang diinputkan, didapatkan Body Performance dengan Grade : ", le.inverse_transform(y_pred3)[0])

    all = st.button("Submit")
    if all :
        submit()
