import joblib
from .preprocess import preprocess_text
from .summarization import abstractive_summary
from gensim import corpora, models

# Load models
clf = joblib.load('c:/Users/g.sneha2/Desktop/nesintel/models/classifier_model.pkl')
tfidf = joblib.load( 'c:/Users/g.sneha2/Desktop/nesintel/models/tfidf_vectorizer.pkl')
lda_model = models.LdaModel.load('c:/Users/g.sneha2/Desktop/nesintel/models/lda_model.gensim')
dictionary = corpora.Dictionary.load('c:/Users/g.sneha2/Desktop/nesintel/models/lda_dictionary.dict')


def predict_category(text):
    clean_text = preprocess_text(text)
    vect_text = tfidf.transform([clean_text])
    pred = clf.predict(vect_text)
    return pred[0]

def get_summary(text):
    return abstractive_summary(text)

def topic_modeling(text, top_n=1):
    clean_tokens = preprocess_text(text).split()
    bow = dictionary.doc2bow(clean_tokens)
    topic_probs = lda_model.get_document_topics(bow)
    
    # Sort by probability and take top topic(s)
    top_topics = sorted(topic_probs, key=lambda x: -x[1])[:top_n]
    
    topics_with_keywords = []
    for topic_id, prob in top_topics:
        keywords = lda_model.show_topic(topic_id, topn=10)
        topics_with_keywords.append((topic_id, keywords))

    return topics_with_keywords







