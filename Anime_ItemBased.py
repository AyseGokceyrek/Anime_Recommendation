import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)


######################################
# Adım 1: Veri Setinin Hazırlanması
######################################

anime = pd.read_csv("C:/Users/Lenovo/PycharmProjects/datasets/Anime/ItemBased/anime.csv")
rating = pd.read_csv("C:/Users/Lenovo/PycharmProjects/datasets/Anime/ItemBased/rating.csv")
anime.head()

rating.head()

"""
Anime.csv
-anime_id - myanimelist.net's unique id identifying an anime.
-name - full name of anime.
-genre - comma separated list of genres for this anime.
-type - movie, TV, OVA, etc.
-episodes - how many episodes in this show. (1 if movie).
-rating - average rating out of 10 for this anime. (members'ların vermiş oldukları puan ortalamalarıdır.)
-members - number of community members that are in this anime's "group". Animeyi izleyenlerin, bitirenlerin, 
yarım bırakanların, drop olanları sayıları toplamını içermektedir.

Rating.csv
-user_id - non identifiable randomly generated user id.
-anime_id - the anime that this user has rated.
-rating - rating out of 10 this user has assigned (-1 if the user watched it but didn't assign a rating).

"""

anime.shape
rating.shape
# Burada çok fazla satır bulunmakta. İngirgeme yaparak daha sağlıklı yorumlamak istiyorum.
# Dolayısı ile anime.csv dosyamızdan gelen members sütunu yardımı ile bu indirgemeyi yapabilirim.
# Ben buradaki değerlerin %75'lik kısmını almak istiyorum. Cünkü, %75'lik kısım demek en fazla izlenen animeler demek şeklinde yorum yapabilirim

a = anime['members'].quantile(0.75)
anime = anime[(anime['members'] >= a)]
anime.shape

rating.loc[rating.rating == -1, 'rating'] = np.NaN
rating.head()

df = anime.merge(rating, how='left', on='anime_id')
df.head()
df.shape


df = df.rename(columns = {"rating_y":"rating"})
df["rating"].isnull().any()


def preprocessor(df):
    df.dropna(inplace=True)
    df.drop("rating_x", axis=1, inplace=True)

preprocessor(df)
df.shape

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################
df.head()
df["name"].nunique()
df["name"].value_counts().head()  # hangi anime kaç rate almış

Rating_Count = pd.DataFrame(df["name"].value_counts())  # Herbir animenin ne kadar rati var onu bulduk.
Rating_Count.shape
rare_anime = Rating_Count[Rating_Count["name"] <=500].index  # 7667 tane anime 500'den az rate sahipmiş
rare_anime

# Az rate alan animelerden kurtulmamız gerekmekte.
common_animes = df[~df["name"].isin(rare_anime)]
common_animes.shape
common_animes["name"].nunique() # Kalan eşsiz animelerimiz
df["name"].nunique()


# Satırlarda kullanıcılar, Sütunlarda isimler olsundu

user_anime_df = common_animes.pivot_table(index=["user_id"], columns=["name"], values="rating")  # burada seyreklik durumu ortaya çıkıyor.
user_anime_df.shape
user_anime_df.head()

"""
User - Anime matrisimiz hazır. Burada iki değişken arasındaki korelasyona bakar gibi bir film ile diğer filmler 
arasındaki korelasyona baktığımızda animelerin benzerliklerini bulabiliriz.
"""

anime_name = "Berserk"
anime_name = "Himouto! Umaru-chan"
anime_name = user_anime_df[anime_name]
anime_name

user_anime_df.corrwith(anime_name).sort_values(ascending=False).head(10)
"""
# Seçmiş olduğumuz animemiz ile diğer animeler arasındaki korelasyona bakıyoruz. 
# Korelasyon değerleri en büyük olan 10 animeyi getirdi.Bunlar bizim önerebildiğimiz animeler
# Burada topluluğun beğenilerle desteklediği, fikir birllikteliğinde bulunduklarını önerdik. İş birlikçi demek de bu demek olmakta.
"""


# Seçtiğimiz animenin dataframe içerisindeki yazım kontrolü için
def check_anime(keyword, user_anime_df):
    return [col for col in user_anime_df.columns if keyword in col]


check_anime("Fairy Tail", user_anime_df)


# Random anime seçimi için

anime_name = pd.Series(user_anime_df.columns).sample(1).values[0]
anime_name = user_anime_df[anime_name]
user_anime_df.corrwith(anime_name).sort_values(ascending=False).head(10)