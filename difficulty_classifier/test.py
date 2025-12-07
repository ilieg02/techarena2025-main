import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os

# ---------------------------------------------------------
# 1. Load your Excel file
# ---------------------------------------------------------
# Make sure your file has columns: "difficulty" and "question"
print(os.system("dir"))
df = pd.read_excel("./inferencePipeline/sample_questions.xlsx")


to_dense = FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)

# ---------------------------------------------------------
# 2. Separate features and labels
# ---------------------------------------------------------
df["difficulty"] = df["difficulty"].replace({
    "medium": "complex",
    "hard": "complex"
})

X = df["question"]
y = df["difficulty"]

# ---------------------------------------------------------
# 3. Train/Test Split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 4. Build the model pipeline
# ---------------------------------------------------------
#model = Pipeline([
#    ("tfidf", TfidfVectorizer()),
#    ("rf", RandomForestClassifier(n_estimators=400, random_state=200, criterion="entropy", oob_score=True))
#])

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),          # unigrams + bigrams
        stop_words="english",        # remove stopwords
        lowercase=True,
        sublinear_tf=True,           # better for text classification
        min_df=2,                    # ignore rare noise words
        max_df=0.9,                  # remove very common non-informative words
        max_features=None,           # let model use all features
        norm="l2",                   # standard for text classification
    )),
    ("rf", RandomForestClassifier(
        n_estimators=800,            # more trees = better accuracy
        random_state=300,
        criterion="entropy",
        oob_score=True,
        class_weight="balanced",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",         # recommended for text RF
        bootstrap=True,              # improves stability
        n_jobs=-1
    ))
])

# ---------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Evaluate the model
# ---------------------------------------------------------
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)
classes = sorted(list(set(y_test)))


cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
print(cm_percent)
# ---------------------------------------------------------
# 7. Try predicting a new question
# ---------------------------------------------------------
example_question = [
    "given a n by n subspace prove that a vector normalization is possible in a k subspaces"
 #   "What is the capital of Japan?"
 #   "what is the height of mount Everest?"
]

prediction = model.predict(example_question)
print("Predicted difficulty:", prediction[0])

# ---------------------------------------------------------
# 8. (Optional) Save the trained model
# ---------------------------------------------------------
import joblib
joblib.dump(model, "difficulty_classifier.pkl")
print("Model saved as difficulty_classifier.pkl")
