import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması


df= pd.read_csv("C:/Users/Lenovo/PycharmProjects/datasets/Anime/animes.csv")

df.head()

# Understanding Data


def df_summary(df):
    print("############### SHAPE ###############")
    print("\n")
    print(df.shape)
    print("############### INDEX ###############")
    print("\n")
    print(df.index)
    print("############### COLUMNS ###############")
    print("\n")
    print(df.columns)
    print("############### DATAFRAME INFORMATIONS ###############")
    print("\n")
    print(df.info())
    print("############### DATAFRAME INFORMATIONS ###############")
    print("\n")
    print(df.describe().T)


df_summary(df)


def missing_value_analysis(df):
    print("Is there any missing value on the dataset?")
    print(df.isnull().values.any())
    missing = df.isnull().values.any()

    if (missing == True):
        print("############### MISSING VALUE COUNTS BY VARIABLES ###############")
        print(df.isnull().sum())
        print("############### TOTAL MISSING VALUE COUNTS ###############")
        print(df.isnull().sum().sum())
    else:
        pass


missing_value_analysis(df)


def preprocessor(df):
    df.dropna(inplace=True)
    df.drop("score", axis=1, inplace=True)


preprocessor(df)
df.shape
#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################


"""
Tf-Idf methodu, kullanacak olduğumuz ingilizce bir metin olduğu için buradaki stop words'ler (and, in, the, on gibi) sil bilgisini gönderiyoruz.
oluşturulacak olan seyrek değerlerin ortaya çıkaracağı problemlerin önüne geçmeye çalışıyoruz ve bunlar ölçüm değerleri taşımamaktadır. On ifadesi 
geçen iki anime birbirine yakın çıkarsa bu b,z, yanıltabilir. 
"""

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["synopsis"])
tfidf_matrix.shape
vectorizer.get_feature_names()
tfidf_matrix.toarray()

#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################
"""
Elimizdeki metinlerin benzerliklerini bulmaya çalışıyoruz. Matematiksel olarak hesaplanacak forma çevirdik. Metin vektörlerini
oluşturduk. Uzaklık temelli ya da benzerlik temelli yaklaşımlar kurularak benzerlikler bulunacak.

"""
cos_sim = cosine_similarity(tfidf_matrix)
cos_sim
cos_sim.shape


def anime_searcher(df, name_words_contain):
    print(df.loc[df["synopsis"].str.contains(name_words_contain, na = False), "title"])


anime_searcher(df, name_words_contain = "Fullmetal Alchemist: Brotherhood")
anime_searcher(df, name_words_contain = "Dragon Ball Z")

#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################


def content_based_recommender(df, anime_name="Bleach", rec_count=10):
    indices = pd.Series(df.index, index = df["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    anime_index = indices[anime_name]
    similarity_score = pd.DataFrame(cos_sim[anime_index], columns = ["score"])
    similar_animes = similarity_score.sort_values(by="score", ascending=False). \
            iloc[1:rec_count].index
    return df["title"].iloc[similar_animes]



content_based_recommender(df,anime_name="Fullmetal Alchemist: Brotherhood")
content_based_recommender(df,anime_name="Dragon Ball Z")

