import numpy as np
import joblib
import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# nltk.download("punkt")
# nltk.download("punkt_tab")

tfidf=joblib.load("models/tfidf.pkl")
category_model=joblib.load("models_SVM/svm_category_model.pkl")
priority_model=joblib.load("models_SVM/svm_priority_model.pkl")
category_encoder=joblib.load("models/category_encoder.pkl")
priority_encoder=joblib.load("models/priority_encoder.pkl")

stop_words=set(stopwords.words("english"))

def summarize_text(text,max_words=20):
    text=text.lower()
    text=re.sub(r"[^a-z\s]","", text)
    text=re.sub(r"\s+", " ", text).strip()
    sentences=sent_tokenize(text)
    if len(sentences)==0:
        return ""
    vectorizer=TfidfVectorizer(stop_words="english")
    tfidf_matrix=vectorizer.fit_transform(sentences)
    sentence_scores=np.array(tfidf_matrix.sum(axis=1)).ravel()
    best_sentence=sentences[np.argmax(sentence_scores)]
    words=[w for w in best_sentence.split() if w not in stop_words]
    summary=" ".join(words[:max_words])
    return summary.capitalize()
def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+|www\S+", "", text)
    text=re.sub(r"[^a-z\s]",'', text)
    text=re.sub(r"\s+", " ", text).strip()
    return text

def softmax(scores):
    exp_scores=np.exp(scores-np.max(scores))
    return exp_scores/exp_scores.sum()


def predict_ticket(subject, description):
    summary=summarize_text(description)
    combined_text=subject+" "+description
    cleaned_text=clean_text(combined_text)
    vec=tfidf.transform([cleaned_text])
    cate_scores=category_model.decision_function(vec)[0]
    cate_probs=softmax(cate_scores)
    cate_pred_idx=np.argmax(cate_probs)
    cate_conf=cate_probs[cate_pred_idx]
    prio_scores=priority_model.decision_function(vec)[0]
    prio_probs=softmax(prio_scores)
    prio_pred_idx=np.argmax(prio_probs)
    prio_conf=prio_probs[prio_pred_idx]
    category=category_encoder.inverse_transform([cate_pred_idx])[0]
    priority=priority_encoder.inverse_transform([prio_pred_idx])[0]
    return {"summary":summary, "category":category,"category_confidence":round(cate_conf,3),"priority":priority,"priority_confidence":round(prio_conf,3)}

if __name__=="__main__":
    subject="Microphone is not working"
    description="I am experiencing an issue with my phone where my voice is not being captured properly during calls, voice messages, or recordings. People on the other end of calls are unable to hear me clearly, and in some cases my voice is completely absent even though the call is connected. The same problem occurs when using voice notes or speech-based features, making communication difficult. Restarting the device and checking permissions have not resolved the issue. This problem started recently and is affecting both personal and professional communication."
    result=predict_ticket(subject,description)
    print("Summary: ",result['summary'])
    print("Category: ",result['category'],"| Confidence: ",result["category_confidence"])
    print("Priority: ",result['priority'],"| Confidence: ",result["priority_confidence"])