import pickle

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("model/classifier.pkl", "rb") as f:
    model = pickle.load(f)


def predict_parksay(text):
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]
    return prob > 0.5, round(prob, 2)
