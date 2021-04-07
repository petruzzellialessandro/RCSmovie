# RCSmovie
This is a package that will be used as `the recommander core` in a telegram bot. 
The core is suggests films using the plots of preferred films of the user. 
So films having similar plots will be suggested.
In this package are implemented 4 main model of spatial vector:
* Doc2Vec (gensim)
* Word2Vec (gensim)
* FastText (gensim and facebook pretrained model)
* TFIDF (gensim)

The suggestions are provided by one of these models. We can choose, and load, the model using the method `select_model(id_model)`.
`id_model` can be:
* 1 to use Doc2Vec. In this case will be use the method `most_similar`.
* 2 to use Doc2Vec. In this case the cosine similarity will be calculated using the centroid.
* 3 to use Word2Vec. In this case will be loaded a pretrained model `word2vec-google-news-300` (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
* 4 to use Word2Vec. In this case will be loaded a model trained with the films plots in dataset.
* 5 to use FastText pretrained. In this case the model is loaded use the file `cc.en.300.bin` (https://fasttext.cc/docs/en/crawl-vectors.html)
* 6 to use FastText. In this case will be loaded a model trained with the films plots in dataset.
* 7 to use TFIDF

The 2 pre-trained have to be downloaded and pasted in the folder `Model\Pretrained`

To get the suggestions we can use the method `get_suggestion(preferences_IDs)`.
`preferences_IDs` is a list of IDs. Each ID matches a film in dataset. This methods return a list of 5 IDs (one per recommended film).