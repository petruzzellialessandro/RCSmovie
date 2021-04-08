import fasttext.util
import numpy as np
import pandas as pd
from gensim.models import FastText
from scipy.spatial import distance


def calculate_centroid(text, model):
    vectors = list()
    if len(text) == 0:
        return [-1]
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


def centroid_fastext_FB(text, model):
    vector_string = list()
    if len(text) == 0:
        return [-1]
    for token in text:
        res = model.get_word_vector(token)
        vector_string.append(res)
        if len(vector_string) == 0:
            vector_string.append(0)
    return np.asarray(vector_string).mean(axis=0)


def create_model_fasttext_fb():
    fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('Models/FastText/cc.en.300.bin')
    return ft


def create_fasttext_model(documents, model_name):
    fasttext = FastText(min_count=1, workers=8, word_ngrams=3, size=100)
    fasttext.build_vocab(documents)
    fasttext.train(documents, total_examples=fasttext.corpus_count, epochs=100)
    fasttext.save(model_name)
    return fasttext


def load_model(documents, model_name):
    try:
        fasttext = FastText.load(model_name)
    except Exception:
        fasttext = create_fasttext_model(documents, model_name)
    return fasttext


def print_res_fastText(token_strings, documents, titles, IDs, modelFastText, pretrained, prefIDs):
    cos_sim_s = []
    if modelFastText is None:
        if pretrained:
            modelFastText = create_model_fasttext_fb()
        else:
            modelFastText = load_model(documents, "FastText/fasttext_model")
    if pretrained:
        querys = list()
        for string in token_strings:
            querys.append(centroid_fastext_FB(string, modelFastText))
        query = np.asarray(querys).mean(axis=0)
    else:
        querys = list()
        for string in token_strings:
            querys.append(calculate_centroid(string, modelFastText))
        query = np.asarray(querys).mean(axis=0)
    for i, doc in enumerate(documents):
        if pretrained:
            films_found = centroid_fastext_FB(doc, modelFastText)
        else:
            films_found = calculate_centroid(doc, modelFastText)
        if films_found[0] == -1:
            cos_sim_s.append(0)
            continue
        cos_sim = 1 - distance.cosine(query, films_found)
        cos_sim_s.append(cos_sim)
    cos_sim_s, titles, IDs = zip(*sorted(zip(cos_sim_s, titles, IDs), reverse=True))
    outputW2V = []
    rank = 1
    for i in range(5+len(token_strings)):
        if len(outputW2V) == 5:
            break
        if prefIDs is not None:
            if IDs[i] in prefIDs:
                continue
        outputW2V.append([rank, titles[i], cos_sim_s[i]])
        rank += 1
    if pretrained:
        print("--------------FastText-PreTrained--------------")
    else:
        print("--------------FastText-Centroid--------------")
    df = pd.DataFrame(outputW2V, columns=["rank", "title", "cosine_similarity"])
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)
