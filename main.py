import csv
import queue
import threading

import gensim.parsing.preprocessing as pp

import Models.Doc2Vec.Doc2Vec as d2v
import Models.Word2Vec.Word2Vec as w2v
import Models.FastText.FastText as ft
import Models.TFIDF.TFIDF as tfidf

global __doc2vec, __most_similar
global __word2vec, __w2c_pre_trained
global __fasttext, __ft_pre_trained
global __tfidf_model, __tfidf_index, __tfidf_dictionary
global __preferences_IDs, __tokenized_plots, __films_IDs, __films_titles
global __id_Model  # il numero che identifica il modello selezionato
global __returned_queue  # returned_queue.get()

__CUSTOM_FILTERS = [lambda x: x.lower(), pp.strip_tags,
                    pp.strip_punctuation,
                    pp.remove_stopwords,
                    pp.split_alphanum,
                    pp.strip_multiple_whitespaces]


def __preprocessing(trama):
    pp_trama = pp.preprocess_string(trama, __CUSTOM_FILTERS)
    return pp_trama


# Questa funzione ci permette di caricare in memoria l'intero dataset. Per sicurezza effettua il processing qualora
# non risulti diviso in token
def __tonkens_from_documents_gensim():
    documents = []
    titles = []
    n_features = []
    pp_docs = []
    IDs = []
    with open('Dataset/tokenFilmsDatset.csv', newline='', encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            tokens = row["Tokens"]
            Title = row["Title"]
            IDs.append(row["ID"])
            titles.append(Title)
            new_tokens = []
            for token in tokens.split(','):
                new_tokens.append(token.replace("""'""", """"""))
            n_features.append(len(new_tokens))
            documents.append(tokens)
            pp_docs.append(__preprocessing(tokens))
    csvfile.close()
    return pp_docs, titles, IDs


def select_model(selected_model):
    global __tokenized_plots, __films_IDs, __films_titles
    global __id_Model
    global __returned_queue
    __tokenized_plots, __films_titles, __films_IDs = __tonkens_from_documents_gensim()

    if selected_model == 1 or selected_model == 2:
        global __doc2vec, __most_similar
        __doc2vec = None
        __returned_queue = queue.Queue()
        thread = threading.Thread(target=d2v.load_model,
                                  args=(__tokenized_plots, "Models/Doc2Vec/doc2vec_model", __returned_queue))
        thread.start()
        if selected_model == 1:
            __most_similar = True
        else:
            __most_similar = False
        __id_Model = selected_model

    if selected_model == 3 or selected_model == 4:
        global __word2vec, __w2c_pre_trained
        __word2vec = None
        if selected_model == 3:
            __w2c_pre_trained = True
        else:
            __w2c_pre_trained = False
        __returned_queue = queue.Queue()
        thread = threading.Thread(target=w2v.load_model, args=(__tokenized_plots, "Models\Word2Vec\word2vec_model",
                                                               __w2c_pre_trained, __returned_queue))
        thread.start()
        __id_Model = selected_model

    if selected_model == 5 or selected_model == 6:
        global __fasttext, __ft_pre_trained
        __fasttext = None
        __returned_queue = queue.Queue()
        if selected_model == 5:
            __ft_pre_trained = True
            thread = threading.Thread(target=ft.create_model_fasttext_fb, args=__returned_queue)
            thread.start()
        else:
            __ft_pre_trained = False
            thread = threading.Thread(target=ft.load_model, args=(__tokenized_plots, "Models/FastText/fasttext_model",
                                                                  __returned_queue))
            thread.start()

    if selected_model == 7:
        global __tfidf_model, __tfidf_index, __tfidf_dictionary
        __tfidf_model = None
        __tfidf_index = None
        __tfidf_dictionary = None
        __returned_queue = queue.Queue()
        thread = threading.Thread(target=tfidf.load_tfidf_model, args=(__tokenized_plots, "Models/TFIDF/tfidf_model",
                                                                       "Models/TFIDF/matrix_tfidf",
                                                                       "Models/TFIDF/dictionary_tfidf",
                                                                       __returned_queue))
        thread.start()

