import pandas as pd
import numpy as np
import re
import nltk
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
# Loading Data
df =pd.read_csv('Data/customer_support_tickets_10k_balanced.csv')
# print(df.head())
# df.info()

#Drop useless
df=df.drop(columns=['ticket_id','customer_name'])
df=df.drop_duplicates()
df=df.dropna()

#Text combination
df['text']=df['subject']+' '+df['description']
df.drop(columns=['subject','description'],inplace=True)

#cleaning 
def clean(text):
    text=text.lower()
    text=re.sub(r'http\S+|www\S+',"",text)
    text=re.sub(r"[^a-z\s]","",text)
    text=re.sub(r"\s+"," ", text).strip()
    return text

df['text']=df['text'].apply(clean)

#nltk
stop_words=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()
df['text']=df['text'].apply(lambda x: " ".join(lemmatizer.lemmatize(w) for w in x.split()if w not in stop_words))

#label
category_encoder=LabelEncoder()
df['category_encoded']=category_encoder.fit_transform(df['category'])
joblib.dump(category_encoder,"models/category_encoder.pkl")

priority_encoder=LabelEncoder()
df['priority_encoded']=priority_encoder.fit_transform(df["priority"])
joblib.dump(priority_encoder,"models/priority_encoder.pkl")

#TF-IDF
tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X=tfidf.fit_transform(df['text'])
joblib.dump(tfidf,'models/tfidf.pkl')
joblib.dump(X,'models/X.pkl')
joblib.dump(df['category_encoded'],'models/y_category.pkl')
joblib.dump(df['priority_encoded'],'models/y_priority.pkl')

print("completed")