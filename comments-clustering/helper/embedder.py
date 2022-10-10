from typing import List, Union

from gensim.models import Word2Vec, FastText
import numpy as np


def text_embeddings(tokenized_corpus: List[List[str]], model: Union[Word2Vec, FastText]) -> np.array:
    """
    Трансформация текстов в векторное представление, путем усреднения 
    входящих в текст векторов слов, полученных на основе модели (gensim Word2Vec)
    tokenized_corpus: список текстов, разбитых на токены
    model: обученная модель (Word2Vec)
    return: nd.array(m, n), где m - количество примеров, n - размерность эмбединга слова  
    """
    n_col = model.vector_size
    n_row = len(tokenized_corpus)
    embeddings = np.zeros(shape=(n_row , n_col))
    for i, sent in enumerate(tokenized_corpus):
        vectors = []
        for word in sent:
            try:
                vectors.append(model.wv[word] )
            except KeyError:   
                continue
        if vectors:
            avg_vec = np.mean(np.array(vectors), axis=0)
            embeddings[i] = avg_vec
    return embeddings 