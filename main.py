import csv
import queue
import threading

import gensim.parsing.preprocessing as pp

import Models.Doc2Vec.Doc2Vec as d2v
import Models.Word2Vec.Word2Vec as w2v
import Models.FastText.FastText as ft
import Models.TFIDF.TFIDF as tfidf

ERROR_FILM_NOT_FOUND = 401

global __doc2vec, __most_similar
global __word2vec, __w2c_pre_trained
global __fasttext, __ft_pre_trained
global __tfidf_model, __tfidf_index, __tfidf_dictionary
global __tokenized_plots, __films_IDs, __films_titles
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


# Questa funzione ci permette di caricare in memoria l'intero dataset dataset diviso in token.
# Questa funzione va sostituita qualora si decidesse di non utilizzare il file già presente.
# Per sicurezza effettua il processing qualora
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


# Funzione esposta che permette di selezionare il modello con cui ottenere risultati. I valori di selected_model sono:
# 1 per usare Doc2Vec con il metodo most_similar.
# 2 per usare Doc2Vec la similarità è con il centroide.
# 3 per usare Word2Vec per utilizzare un modello pre-addestrato (word2vec-google-news-300).
# 4 per usare Word2Vec.
# 5 per usare FastText per utilizzare un modello pre-addestrato (cc.en.300.bin).
# 6 per usare FastText.
# 7 per usare TFIDF
def select_model(selected_model):
    global __tokenized_plots, __films_IDs, __films_titles
    global __id_Model
    global __returned_queue
    try:
        if __tokenized_plots is not None and __films_IDs is not None and __films_titles is not None:
            print("Movie Info Already Loaded") #Già caricati in memoria le informazini sui film
        else:
            raise Exception
    except Exception:
        __tokenized_plots, __films_titles, __films_IDs = __tonkens_from_documents_gensim()
    #Selezione del modello DOC2VEC
    if selected_model == 1 or selected_model == 2:
        global __doc2vec, __most_similar
        try:
            if __doc2vec is not None:
                print("Already Loaded") #Si evita di ricaricare il modello
            else:
                raise Exception
        except Exception:
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
    #Selezione del modello WORD2VEC
    if selected_model == 3 or selected_model == 4:
        global __word2vec, __w2c_pre_trained
        if selected_model == 3:
            __w2c_pre_trained = True
        else:
            __w2c_pre_trained = False
        try:
            if __word2vec is not None and __id_Model == selected_model:
                print("Already Loaded")#Si evita di ricaricare il modello
            else:
                raise Exception
        except Exception:
            __word2vec = None
            __returned_queue = queue.Queue()
            thread = threading.Thread(target=w2v.load_model, args=(__tokenized_plots, "Models\Word2Vec\word2vec_model",
                                                                   __w2c_pre_trained, __returned_queue))
            thread.start()
        __id_Model = selected_model
    #Selezione del modello FASTTEXT
    if selected_model == 5 or selected_model == 6:
        global __fasttext, __ft_pre_trained
        __returned_queue = queue.Queue()
        if selected_model == 5:
            __ft_pre_trained = True
            try:
                if __fasttext is not None and selected_model == __id_Model:
                    print("Already Loaded")#Si evita di ricaricare il modello
                else:
                    raise Exception
            except Exception:
                __fasttext = None
                thread = threading.Thread(target=ft.create_model_fasttext_fb, args=(__fasttext, __returned_queue))
                thread.start()
        else:
            __ft_pre_trained = False
            try:
                if __fasttext is not None and selected_model == __id_Model:
                    print("Already Loaded")
                else:
                    raise Exception
            except Exception:
                __fasttext = None
                thread = threading.Thread(target=ft.load_model, args=(__tokenized_plots, "Models/FastText"
                                                                                         "/fasttext_model",
                                                                      __returned_queue))
                thread.start()
        __id_Model = selected_model
    # Selezione del modello TFIDF
    if selected_model == 7:
        global __tfidf_model, __tfidf_index, __tfidf_dictionary
        try:
            if __tfidf_model is not None and __tfidf_index is not None and __tfidf_dictionary is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __tfidf_model = None
            __tfidf_index = None
            __tfidf_dictionary = None
            __returned_queue = queue.Queue()
            thread = threading.Thread(target=tfidf.load_tfidf_model,
                                      args=(__tokenized_plots, "Models/TFIDF/tfidf_model",
                                            "Models/TFIDF/matrix_tfidf",
                                            "Models/TFIDF/dictionary_tfidf",
                                            __returned_queue))
            thread.start()
        __id_Model = selected_model
    else:
        return 404 #MODELLO NON TROVATO


# Funzionalità esposta. In input c'è una lista di IDs per cui è presente una preferenza.
# NB: Anche se è una preferenza, deve essere una lista.
# In caso non sia sata chiamata la funzione select_model() allora __film_IDs è null quindi si sollega l'eccezione per
# cui sarà restituito "ERROR_FILM_NOT_FOUND"
def get_suggestion(preferences_IDs):
    global __tokenized_plots, __films_IDs, __films_titles
    IDs_pref = list()
    tokenized_pref = list()
    for id in preferences_IDs:
        try:
            index = __films_IDs.index(id)
            IDs_pref.append(id)
            tokenized_pref.append(__tokenized_plots[index])
            recommends = __get_rec(IDs_pref, tokenized_pref)
            return recommends
        except Exception:
            return ERROR_FILM_NOT_FOUND


# Funzione che effettivamente si occupa di generare le raccomandazioni in base al modello.
# DA NON CHIAMARE
def __get_rec(IDs_pref, tokenized_pref):
    global __tokenized_plots, __films_IDs, __films_titles
    global __id_Model
    global __returned_queue
    if __id_Model == 1 or __id_Model == 2:
        global __doc2vec, __most_similar
        try:
            if __doc2vec is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __doc2vec = __returned_queue.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = d2v.print_res_doc2vec(token_strings=tokenized_pref, documents=__tokenized_plots,
                                           titles=__films_titles, IDs=__films_IDs, modelDoC=__doc2vec,
                                           most_similar=__most_similar, prefIDs=IDs_pref)
    elif __id_Model == 3 or __id_Model == 4:
        global __word2vec, __w2c_pre_trained
        try:
            if __word2vec is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __word2vec = __returned_queue.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = w2v.print_res_word2vec(token_strings=tokenized_pref, documents=__tokenized_plots,
                                            titles=__films_titles, IDs=__films_IDs, modelWord=__word2vec,
                                            pretrained=__w2c_pre_trained, prefIDs=IDs_pref)
    elif __id_Model == 5 or __id_Model == 6:
        global __fasttext, __ft_pre_trained
        try:
            if __fasttext is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __fasttext = __returned_queue.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = ft.print_res_fastText(token_strings=tokenized_pref, documents=__tokenized_plots,
                                           titles=__films_titles, IDs=__films_IDs, modelFastText=__fasttext,
                                           pretrained=__ft_pre_trained, prefIDs=IDs_pref)
    elif __id_Model == 7:
        global __tfidf_model, __tfidf_index, __tfidf_dictionary
        try:
            if __tfidf_model is not None and __tfidf_index is not None and __tfidf_dictionary is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            loaded = __returned_queue.get()
            __tfidf_model = loaded[0]
            __tfidf_index = loaded[1]
            __tfidf_dictionary = loaded[2]
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = tfidf.print_res_tfidf(token_strings=tokenized_pref, documents=__tokenized_plots,
                                           titles=__films_titles, IDs=__films_IDs, dictionary=__tfidf_dictionary,
                                           tfidfmodel=__tfidf_model, index=__tfidf_index, prefIDs=IDs_pref)
    return recommends

