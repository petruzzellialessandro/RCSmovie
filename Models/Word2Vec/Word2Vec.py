import pandas as pd
from gensim.models import Word2Vec
from gensim import models
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


def create_model_word2vec(documents, model_name):
    word2vec = Word2Vec(size=100, min_count=1, workers=8)
    word2vec.build_vocab(documents)
    word2vec.train(documents, epochs=100, total_examples=word2vec.corpus_count)
    word2vec.save(model_name)
    return word2vec


def load_model(documents, model_name, pretrained):
    if pretrained:
        word2vec = models.KeyedVectors.load_word2vec_format('Models/Word2Vec/word2vec-google-news-300.bin', binary=True)
        return word2vec
    try:
        word2vec = Word2Vec.load(model_name)
    except Exception:
        word2vec = create_model_word2vec(documents, model_name)
    return word2vec


def print_res_word2vec(token_strings, documents, titles, IDs, pretrained, modelWord, prefIDs):
    if modelWord is None:
        modelWord = load_model(documents, "Word2Vec\word2vec_model", pretrained)
    cos_sim_s = []
    querys = list()
    for string in token_strings:
        querys.append(calculate_centroid(string, modelWord))
    query = np.asarray(querys).mean(axis=0)
    for i, doc in enumerate(documents):
        films_found = calculate_centroid(doc, modelWord)
        if len(films_found) == 0:
            cos_sim_s.append(0)
            continue
        cos_sim = 1 - distance.cosine(query, films_found)
        cos_sim_s.append(cos_sim)
    cos_sim_s, titles, IDs = zip(*sorted(zip(cos_sim_s, titles, IDs), reverse=True))
    outputW2V = []
    rank = 1
    for i in range(5 + len(token_strings)):
        if len(outputW2V) == 5:
            break
        if prefIDs is not None:
            if IDs[i] in prefIDs:
                continue
        outputW2V.append([rank, titles[i], cos_sim_s[i]])
        rank += 1
    if pretrained:
        print("--------------W2V-PreTrained--------------")
    else:
        print("--------------W2V-Centroid--------------")
    df = pd.DataFrame(outputW2V, columns=["rank", "title", "cosine_similarity"])
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
