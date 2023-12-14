# import library
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Tambahkan SVM
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
	page_title='Analisis Sentimen App',  # String or None. Strings get appended with "• Streamlit". 
	page_icon=':face_with_monocle:',  # String, anything supported by st.image, or None.
)
# Judul Aplikasi
st.title("Analisis Sentimen App")


# Upload dataset
uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Membaca dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    # profile = ProfileReport(df, title="Profiling Report")


    # Menampilkan beberapa baris pertama dari dataset
    st.write("Dataset:")
    st.dataframe(df)
    # st.write(df.head())

    # Pilih kolom teks
    text_column = st.selectbox("Pilih Kolom Teks", df.columns)

    # Pilih kolom label
    label_column = st.selectbox("Pilih Kolom Label", df.columns)

     # Word Cloud untuk teks
    st.header("Data Visualization")
    st.subheader("Wordcloud")
    if not df[text_column].empty:
        wordcloud = WordCloud().generate(" ".join(df[text_column]))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

    # Diagram Pie untuk label
    st.subheader("Diagram Pie")
    if not df[label_column].empty:
        label_counts = df[label_column].value_counts()
        fig_pie = go.Figure(data=[go.Pie(labels=label_counts.index, values=label_counts)])
        st.plotly_chart(fig_pie)

    # Pilih model machine learning
    model_option = st.selectbox("Pilih Model", ["Multinomial Naive Bayes", "Random Forest", "Logistic Regression", "SVM"])

    # Pemrosesan data sederhana
    # Label Encoding kolom label
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

   

    # Menghapus stop words menggunakan Sastrawi
    stopword_factory = StopWordRemoverFactory()
    stopword = stopword_factory.create_stop_word_remover()
    df[text_column] = df[text_column].apply(lambda x: stopword.remove(x))

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[text_column])
    y = df[label_column]

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat model
    if model_option == "Multinomial Naive Bayes":
        model = MultinomialNB()
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    elif model_option == "SVM":
        model = SVC()

    # Melatih model
    model.fit(X_train, y_train)

    # Melakukan prediksi
    y_pred = model.predict(X_test)

    # Menampilkan metrik evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy Score: {accuracy * 100:.2f}%")


    st.header('Try Prediction Text')
    input_text = st.text_input("Masukkan teks:")

    # Tombol prediksi
    if st.button("Prediksi"):
        # Prediksi hanya jika tombol ditekan
        if input_text is not None and input_text != "":
            input_vector = vectorizer.transform([input_text])
            prediction = model.predict(input_vector)
            label = label_encoder.inverse_transform(prediction)[0]
            st.write("Prediksi:", label)
        else:
            # Tampilkan pesan jika input kosong
            st.warning("Masukkan teks untuk prediksi.")
        
        
