
##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################
df = sns.load_dataset("titanic")
df.shape
df.info()
df.head()

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################
df["sex"].value_counts()
#male      577
#female    314

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################
df.nunique()
df.nunique().to_frame().T
#   survived  pclass  sex  age  sibsp  parch  fare  embarked  class  who  adult_male  deck  embark_town  alive  alone
#      2       3       2   88     7     7      248     3        3     3        2        7        3         2      2


#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################
df["pclass"].unique()
#3
df["pclass"].nunique()
#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################
df[["pclass", "parch"]].nunique()
#pclass    3
#parch     7

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################
df["embarked"].dtype
# <StringDtype(na_value=nan)>
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype
#CategoricalDtype(categories=['C', 'Q', 'S'], ordered=False, categories_dtype=str)

#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################
df[df["embarked"] == "C"].head()

df["embarked"] == "C" # embarked sutununda C olanlar gelır

df[df["embarked"] == "C"] # tum veride embarked sutununda C olanlar gelir

df[df["embarked"] == "C"][:10] # 10 tanesı gelir

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################
df[df["embarked"] != "S"].head()
df[df["embarked"] != "S"][:10]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################
df[(df["age"] < 30) & (df["sex"] == "female")].head()

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################
df[(df["fare"] > 500) | (df["age"] > 70)].head()


#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################
df.isnull().sum()


#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################
df.drop(columns="who").head()
df.head()

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################
deck_mode = df["deck"].mode()[0]
df["deck"] = df["deck"].fillna(deck_mode)
deck_mode

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################
df.isnull().sum()
df["age"].median()
df["age"] = df["age"].fillna(age_median)
df["age"].isnull().sum()


#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################
df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})
# tersi
df.groupby(["sex", "pclass"]).agg({"survived": ["sum", "count", "mean"]})

# 2. YOL BONUS
### istatistikler bır tabloya cevırelım
stats = ["sum", "count", "mean"]
grouped_df = df.groupby(["pclass", "sex"])["survived"].agg(stats)
grouped_df

grouped_df.columns
[f"survived_{stat}" for stat in stats]

grouped_df.columns = [f"survived_{stat}" for stat in stats]
grouped_df.head()
grouped_df = pd.DataFrame(grouped_df).reset_index() # ındek sıfırlansın
grouped_df.head()
#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################
def age_classifier(age):
    return 1 if age < 30 else 0

df["age_flag"] = df["age"].apply(lambda x: age_classifier(x))


#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################
import seaborn as sns
df_tips = sns.load_dataset("tips")
df_tips.head()



#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df_tips.groupby("time")["total_bill"].agg(["sum", "min", "max", "mean"])


#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################
df_tips.groupby(["day", "time"])["total_bill"].agg(["sum", "min", "max", "mean"])

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################
# time = lunch & sex = female

# once koşul sonra kırılım sonra istatistikler

df_tips[(df_tips["time"] == "Lunch") & (df_tips["sex"] == "Female")].head()

df_tips[(df_tips["time"] == "Lunch") & (df_tips["sex"] == "Female")].groupby("day").agg({
    "total_bill": ["sum", "min", "max", "mean"],
    "tip": ["sum", "min", "max", "mean"]
})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################
df_tips.loc[(df_tips["size"] < 3) & (df_tips["total_bill"] > 10), "total_bill"].mean()


#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################
df_tips["total_bill_tip_sum"] = df_tips["total_bill"] + df_tips["tip"]



#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################
top30_df = df_tips.sort_values("total_bill_tip_sum", ascending=False).head(30)
top30_df


