import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess_text
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

#import the data
data = pd.read_csv('../data/news.csv')

#preprocess the text column
data['clean_text'] = data['text'].apply(preprocess_text)

#define x and y
x = data['clean_text']
y = data['labels']

#vectorize the text
tfidfmodel = TfidfVectorizer(max_features=10000,ngram_range=(1, 3),min_df=2,max_df=0.95)
x_vect = tfidfmodel.fit_transform(x)

#split the data into train and test
x_train , x_test , y_train , y_test = train_test_split(x_vect , y , stratify=y,test_size=0.2 , random_state=42)

#model training
model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(x_train , y_train)
y_pred = model_logistic.predict(x_test)


#saving the models to pickle files
joblib.dump(model_logistic, '../models/classifier_model.pkl')
joblib.dump(tfidfmodel, '../models/tfidf_vectorizer.pkl')





