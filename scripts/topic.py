import pandas as pd
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from preprocess import preprocess_text


def train_lda_model(documents, num_topics=10):
    # Preprocess documents
    processed_docs = [preprocess_text(doc).split() for doc in documents]
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    # Train LDA model
    lda_model = models.LdaModel(corpus=corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                passes=10,
                                random_state=42)
    # Save the model and dictionary
    lda_model.save('../models/lda_model.gensim')
    dictionary.save('../models/lda_dictionary.dict')
    return lda_model, corpus, dictionary

def get_topic_data(lda_model, corpus, dictionary, topn=10):
    topic_keywords = {
        topic_id: [word for word, _ in lda_model.show_topic(topic_id, topn=topn)]
        for topic_id in range(lda_model.num_topics)
    }

    topic_sizes = [0] * lda_model.num_topics
    topic_probs = [0.0] * lda_model.num_topics
    total_docs = len(corpus)

    for doc_bow in corpus:
        topic_dist = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
        for topic_id, prob in topic_dist:
            topic_sizes[topic_id] += 1
            topic_probs[topic_id] += prob

    topic_data = [
        (topic_id, topic_keywords[topic_id], topic_sizes[topic_id], topic_probs[topic_id] / total_docs)
        for topic_id in range(lda_model.num_topics)
    ]
    return topic_data


if __name__ == "__main__":
    df = pd.read_csv('../data/news.csv')
    documents = df['text'].dropna().tolist()
    model, corpus, dictionary = train_lda_model(documents, num_topics=10)
    topic_data = get_topic_data(model, corpus, dictionary)
    df_topics = pd.DataFrame([
        {
            "TopicID": tid,
            "Keywords": ", ".join(keywords),
            "Document Count": size,
            "Average Probability": round(prob, 4)
        }
        for tid, keywords, size, prob in topic_data
    ])
    
    print(df_topics)
    df_topics.to_csv('../data/lda_topic_summary.csv', index=False)

