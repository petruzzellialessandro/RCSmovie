import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.spatial import distance
import numpy as np


def calculate_centroid(text, model):
    vectors = list()
    for word in text:
        try:
            vector = model[word]
            vectors.append(vector)
        except Exception:
            # print(word, " non c'è")
            continue
    if vectors:
        return np.asarray(vectors).mean(axis=0)
    return np.array([])


def create_model_doc2vec(documents, model_name):
    tagged_documents = []
    for i, doc in enumerate(documents):
        tagged_documents.append(TaggedDocument(doc, [i]))
    doc2vec = Doc2Vec(tagged_documents, vector_size=100, min_count=3, workers=8, epochs=100)
    # doc2vec.build_vocab(tagged_documents)
    doc2vec.train(tagged_documents, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)
    doc2vec.save(model_name)
    return doc2vec


def load_model(documents, model_name, queue=None):
    try:
        model = Doc2Vec.load(model_name)
    except Exception:
        model = create_model_doc2vec(documents, model_name)
    if queue is not None:
        queue.put(model)
    return model


def get_recommendations_doc2vec(token_strings, documents, titles, IDs, modelDoC, most_similar, prefIDs):
    recommend_movies = []
    num_recommends = 5
    cos_sim_s = []
    if modelDoC is None:
        modelDoC = load_model(documents, "Models/Doc2Vec/doc2vec_model", None)
    if most_similar:
        queries = list()
        try:
            for string in token_strings:
                new_sentence_vectorized = modelDoC.infer_vector(string, steps=100)
                queries.append(new_sentence_vectorized)
        except Exception:
            new_sentence_vectorized = modelDoC.infer_vector(token_strings, steps=100)
            queries.append(new_sentence_vectorized)
        query = np.asarray(queries).mean(axis=0)
        similar_sentences = modelDoC.docvecs.most_similar([query], topn=5 + len(token_strings))
        outputD2V_MS = []
        rank = 1
        for i, v in enumerate(similar_sentences):
            index = v[0]
            if len(outputD2V_MS) == num_recommends:
                break
            if prefIDs is not None:
                if IDs[index] in prefIDs:
                    continue
            outputD2V_MS.append([rank, titles[index], v[1]])
            recommend_movies.append({"Rank": rank, "ID": IDs[index]})
            rank += 1
        # print("--------------D2V-most_similar--------------")
        # df = pd.DataFrame(outputD2V_MS, columns=["rank", "title", "cosine_similarity"])
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        # print(df)
    else:
        queries = list()
        for string in token_strings:
            queries.append(calculate_centroid(string, modelDoC))
        query = np.asarray(queries).mean(axis=0)
        for i, doc in enumerate(documents):
            films_found = calculate_centroid(doc, modelDoC)
            if len(films_found) == 0:
                cos_sim_s.append(0)
                continue
            try:
                cos_sim = 1 - distance.cosine(query, films_found)
            except Exception:
                cos_sim = 0
            cos_sim_s.append(cos_sim)
        cos_sim_s, titles, IDs = zip(*sorted(zip(cos_sim_s, titles, IDs), reverse=True))
        outputD2VCent = []
        rank = 1
        for i in range(num_recommends + len(token_strings)):
            if len(outputD2VCent) == num_recommends:
                break
            if prefIDs is not None:
                if IDs[i] in prefIDs:
                    continue
            outputD2VCent.append([rank, titles[i], cos_sim_s[i]])
            recommend_movies.append({"Rank": rank, "ID": IDs[i]})
            rank += 1
        # print("--------------D2V-Centroid--------------")
        # df = pd.DataFrame(outputD2VCent, columns=["rank", "title", "cosine_similarity"])
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        # print(df)
    return recommend_movies
