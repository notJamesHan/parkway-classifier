import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


df1 = pd.read_csv("data/parksay.tsv", sep="\t")  # label = 1
df2 = pd.read_csv("data/ChatbotData.csv")  # label = 0

# Clean up chatbot data
# we can technically use Q column and A column
df2 = df2[["Q", "A"]]
# Combine Q and A into one text column
df2["text"] = df2["Q"] + " " + df2["A"]
df2 = df2.drop(columns=["Q", "A"])  # Remove the original columns
df2 = df2.dropna()
df2 = df2[df2["text"].str.len() > 0]
df2 = df2.sample(n=120, random_state=42)  # reduce size for faster training

df1["label"] = 1
df2["label"] = 0
df = pd.concat([df1, df2])

X = df["text"]
y = df["label"]
tfidf = TfidfVectorizer(max_features=3000)
X_vec = tfidf.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)

# 저장
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("model/classifier.pkl", "wb") as f:
    pickle.dump(model, f)
