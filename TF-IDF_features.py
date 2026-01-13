import joblib
import pandas as pd
X=joblib.load("models/X.pkl")
tfidf=joblib.load("models/tfidf.pkl")
feature_names=pd.DataFrame({"word":tfidf.get_feature_names_out(),"idf":tfidf.idf_})
feature_names=feature_names.sort_values(by="idf", ascending=True)
print("TF-IDF Shape: ",X.shape)
print("Type: ",type(X))
print("Total Features: ",len(feature_names))
print(feature_names.head(10))