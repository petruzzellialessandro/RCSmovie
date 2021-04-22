import csv
import queue
import threading

import gensim.parsing.preprocessing as __pp__

import Models.Doc2Vec.Doc2Vec as __d2v__
import Models.Word2Vec.Word2Vec as __w2v__
import Models.FastText.FastText as __ft__
import Models.TFIDF.TFIDF as __tfidf__

global __doc2vec__, __most_similar__
global __word2vec__, __w2c_pre_trained__
global __fasttext__, __ft_pre_trained__
global __tfidf_model__, __tfidf_index__, __tfidf_dictionary__
global __tokenized_plots__, __films_IDs__, __films_titles__
global __id_Model__  # il numero che identifica il modello selezionato
global __returned_queue__  # returned_queue.get()

__CUSTOM_FILTERS__ = [lambda x: x.lower(), __pp__.strip_tags,
                      __pp__.strip_punctuation,
                      __pp__.remove_stopwords,
                      __pp__.split_alphanum,
                      __pp__.strip_multiple_whitespaces]


def __preprocessing__(trama):
    pp_trama = __pp__.preprocess_string(trama, __CUSTOM_FILTERS__)
    return pp_trama


def __update_file__(index, append):
    global __tokenized_plots__, __films_IDs__, __films_titles__
    if append:
        with open('Dataset/tokenFilmsDatset.csv', "a", newline='', encoding="utf8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([__films_IDs__[index], __films_titles__[index], __tokenized_plots__[index]])
            csvfile.close()
            return 200
    with open('Dataset/tokenFilmsDatset.csv', "r+", newline='', encoding="utf8") as csvfile:
        fieldnames = ['ID', 'Title', "Tokens"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, (ID, title, plot) in enumerate(zip(__films_IDs__, __films_titles__, __tokenized_plots__)):
            writer.writerow({"ID": ID, "Title": title, "Tokens": plot})
            if i > index:
                csvfile.close()
                return 200
    csvfile.close()


# Questa funzione ci permette di caricare in memoria l'intero dataset dataset diviso in token.
# Questa funzione va sostituita qualora si decidesse di non utilizzare il file già presente.
# Per sicurezza effettua il processing qualora
# non risulti diviso in token
def __tonkens_from_documents_gensim__():
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
            pp_docs.append(__preprocessing__(tokens))
    csvfile.close()
    return pp_docs, titles, IDs


# Funzione esposta che permette di selezionare il modello con cui ottenere risultati. I valori di selected_model sono:
# 1 per usare Doc2Vec con il metodo most_similar.
# 2 per usare Doc2Vec la similarità è con il centroide.
# 3 per usare Word2Vec per utilizzare un modello pre-addestrato (word2vec-google-news-300).
# 4 per usare Word2Vec.
# 5 per usare FastText per utilizzare un modello pre-addestrato (cc.en.300.bin).
# 6 per usare FastText.
# 7 per usare tfidf.
def select_model(selected_model):
    global __tokenized_plots__, __films_IDs__, __films_titles__
    global __id_Model__
    global __returned_queue__
    try:
        if __tokenized_plots__ is not None and __films_IDs__ is not None and __films_titles__ is not None:
            print("Movie Info Already Loaded")  # Già caricati in memoria le informazioni sui film
        else:
            raise Exception
    except Exception:
        __tokenized_plots__, __films_titles__, __films_IDs__ = __tonkens_from_documents_gensim__()
    # Selezione del modello DOC2VEC
    if selected_model == 1 or selected_model == 2:
        global __doc2vec__, __most_similar__
        try:
            if __doc2vec__ is not None:
                print("Already Loaded")  # Si evita di ricaricare il modello
            else:
                raise Exception
        except Exception:
            __doc2vec__ = None
            __returned_queue__ = queue.Queue()
            thread = threading.Thread(target=__d2v__.load_model,
                                      args=(__tokenized_plots__, "Models/Doc2Vec/doc2vec_model", __returned_queue__))
            thread.start()
        if selected_model == 1:
            __most_similar__ = True
        else:
            __most_similar__ = False
        __id_Model__ = selected_model
        return 200
    # Selezione del modello WORD2VEC
    if selected_model == 3 or selected_model == 4:
        global __word2vec__, __w2c_pre_trained__
        if selected_model == 3:
            __w2c_pre_trained__ = True
        else:
            __w2c_pre_trained__ = False
        try:
            if __word2vec__ is not None and __id_Model__ == selected_model:
                print("Already Loaded")  # Si evita di ricaricare il modello
            else:
                raise Exception
        except Exception:
            __word2vec__ = None
            __returned_queue__ = queue.Queue()
            thread = threading.Thread(target=__w2v__.load_model,
                                      args=(__tokenized_plots__, "Models\Word2Vec\word2vec_model",
                                            __w2c_pre_trained__, __returned_queue__))
            thread.start()
        __id_Model__ = selected_model
        return 200
    # Selezione del modello FASTTEXT
    if selected_model == 5 or selected_model == 6:
        global __fasttext__, __ft_pre_trained__
        __returned_queue__ = queue.Queue()
        if selected_model == 5:
            __ft_pre_trained__ = True
            try:
                if __fasttext__ is not None and selected_model == __id_Model__:
                    print("Already Loaded")  # Si evita di ricaricare il modello
                else:
                    raise Exception
            except Exception:
                __fasttext__ = None
                thread = threading.Thread(target=__ft__.create_model_fasttext_fb,
                                          args=(__fasttext__, __returned_queue__))
                thread.start()
        else:
            __ft_pre_trained__ = False
            try:
                if __fasttext__ is not None and selected_model == __id_Model__:
                    print("Already Loaded")
                else:
                    raise Exception
            except Exception:
                __fasttext__ = None
                thread = threading.Thread(target=__ft__.load_model, args=(__tokenized_plots__, "Models/FastText"
                                                                                               "/fasttext_model",
                                                                          __returned_queue__))
                thread.start()
        __id_Model__ = selected_model
        return 200
    # Selezione del modello __tfidf__
    if selected_model == 7:
        global __tfidf_model__, __tfidf_index__, __tfidf_dictionary__
        try:
            if __tfidf_model__ is not None and __tfidf_index__ is not None and __tfidf_dictionary__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __tfidf_model__ = None
            __tfidf_index__ = None
            __tfidf_dictionary__ = None
            __returned_queue__ = queue.Queue()
            thread = threading.Thread(target=__tfidf__.load_model,
                                      args=(__tokenized_plots__, "Models/TFIDF/tfidf_model",
                                            "Models/TFIDF/matrix_tfidf",
                                            "Models/TFIDF/dictionary_tfidf",
                                            __returned_queue__))
            thread.start()
        __id_Model__ = selected_model
        return 200
    else:
        return 404  # MODELLO NON TROVATO


# Funzionalità esposta. In input c'è una lista di IDs per cui è presente una preferenza.
# NB: Anche se è una preferenza, deve essere una lista.
# In caso non sia sata chiamata la funzione select_model() allora __film_IDs è null quindi si sollega l'eccezione per
# cui sarà restituito "ERROR_FILM_NOT_FOUND"
def get_suggestion(preferences_IDs):
    global __tokenized_plots__, __films_IDs__, __films_titles__
    IDs_pref = list()
    tokenized_pref = list()
    for id in preferences_IDs:
        try:
            index = __films_IDs__.index(id)
            IDs_pref.append(id)
            tokenized_pref.append(__tokenized_plots__[index])
            recommends = __get_rec__(IDs_pref, tokenized_pref)
            return recommends
        except Exception:
            return 400 #Film non trovato


# Funzione che effettivamente si occupa di generare le raccomandazioni in base al modello.
# DA NON CHIAMARE
def __get_rec__(IDs_pref, tokenized_pref):
    global __tokenized_plots__, __films_IDs__, __films_titles__
    global __id_Model__
    global __returned_queue__
    if __id_Model__ == 1 or __id_Model__ == 2:
        global __doc2vec__, __most_similar__
        try:
            if __doc2vec__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __doc2vec__ = __returned_queue__.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __d2v__.get_recommendations_doc2vec(token_strings=tokenized_pref, documents=__tokenized_plots__,
                                                         titles=__films_titles__, IDs=__films_IDs__,
                                                         modelDoC=__doc2vec__,
                                                         most_similar=__most_similar__, prefIDs=IDs_pref)
    elif __id_Model__ == 3 or __id_Model__ == 4:
        global __word2vec__, __w2c_pre_trained__
        try:
            if __word2vec__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __word2vec__ = __returned_queue__.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __w2v__.get_recommendations_word2vec(token_strings=tokenized_pref, documents=__tokenized_plots__,
                                                          titles=__films_titles__, IDs=__films_IDs__,
                                                          modelWord=__word2vec__,
                                                          pretrained=__w2c_pre_trained__, prefIDs=IDs_pref)
    elif __id_Model__ == 5 or __id_Model__ == 6:
        global __fasttext__, __ft_pre_trained__
        try:
            if __fasttext__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            __fasttext__ = __returned_queue__.get()
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __ft__.get_recommendations_fastText(token_strings=tokenized_pref, documents=__tokenized_plots__,
                                                         titles=__films_titles__, IDs=__films_IDs__,
                                                         modelFastText=__fasttext__,
                                                         pretrained=__ft_pre_trained__, prefIDs=IDs_pref)
    elif __id_Model__ == 7:
        global __tfidf_model__, __tfidf_index__, __tfidf_dictionary__
        try:
            if __tfidf_model__ is not None and __tfidf_index__ is not None and __tfidf_dictionary__ is not None:
                print("Already Loaded")
            else:
                raise Exception
        except Exception:
            loaded = __returned_queue__.get()
            __tfidf_model__ = loaded[0]
            __tfidf_index__ = loaded[1]
            __tfidf_dictionary__ = loaded[2]
            # Se il modello non è caricato lo prendiamo dalla cosa dei risultato. Aspetta che termini il thread
        recommends = __tfidf__.get_recommendations_tfidf(token_strings=tokenized_pref, documents=__tokenized_plots__,
                                                         titles=__films_titles__, IDs=__films_IDs__,
                                                         dictionary=__tfidf_dictionary__,
                                                         tfidfmodel=__tfidf_model__, index=__tfidf_index__,
                                                         prefIDs=IDs_pref)
    return recommends


def update_dataset(ID, title, plot):
    append = True
    global __tokenized_plots__, __films_IDs__, __films_titles__
    try:
        if __tokenized_plots__ is not None and __films_IDs__ is not None and __films_titles__ is not None:
            print("Movie Info Already Loaded")  # Già caricati in memoria le informazioni sui film
        else:
            raise Exception
    except Exception:
        __tokenized_plots__, __films_titles__, __films_IDs__ = __tonkens_from_documents_gensim__()
    try:
        index = __films_IDs__.index(ID)
        __films_IDs__.remove(ID)
        __films_titles__.remove(__films_titles__[index])
        __tokenized_plots__.remove(__tokenized_plots__[index])
        append = False
    except ValueError:
        index = len(__films_IDs__)
    __tokenized_plots__.insert(index, __preprocessing__(plot))
    __films_IDs__.insert(index, ID)
    __films_titles__.insert(index, title)
    if __update_file__(index, append) == 200:
        return 200
    else:
        return 400  #file non aggiornato


def get_suggestion_from_sentence(sentence):
    try:
        recommends = __get_rec__(None, __preprocessing__(sentence))
        return recommends
    except Exception:
        return 400


