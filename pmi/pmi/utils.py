from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
from nltk.tokenize import RegexpTokenizer
from scipy.sparse import csr_matrix


def text_tokenize(corpus: List[str], min_symbols: int = 2, min_tokens: int = 3,
                  stopwords=Optional[set], regexp: str = r'[\w\d]+') -> List[List[str]]:
    """
    Токенизация + фильтрация текста
    :param corpus: корпус текстов
    :param min_symbols: минимальное кол-во символов для токена
    :param min_tokens: минимальное количество токенов для текста
    :param stopwords: множество стоп-слов
    :param regexp: регулярное выражение для поиска токенов
    :return: токенизированный текст
    """
    if stopwords is None:
        stopwords = set()
    tokenizer = RegexpTokenizer(regexp)
    text_tokenized = [tokenizer.tokenize(text.lower()) for text in corpus]
    text_tokenized = [[token for token in text if token not in stopwords and len(token) >= min_symbols] 
                      for text in text_tokenized]
    text_tokenized = [text for text in text_tokenized if len(text) >= min_tokens]
    return text_tokenized


def print_minmax(matrix, title=''):
    """Распечатать минимум и макимум матрицы"""
    print(f'{title}: min={matrix.min():.3f}, max={matrix.max():.3f}')


def print_nnz_sparse(sparse_matrix, title=''):
    """Распечатать кол-во ненулевых значений разряженной матрицы в процентах"""
    print('({}) Ненулевых значений: {:.2f} %.'.format(title, 
          sparse_matrix.nnz * 100 / (sparse_matrix.shape[0] * sparse_matrix.shape[1])))


def plot_weights(data: list, name: list, log: bool = True):
    """Отрисовать гистограммы для признаков"""
    fig, axes = plt.subplots(1, len(data), figsize=(12, 5))
    plt.suptitle('Гистограмма признаков (весов слов)')
    for i, (X, n) in enumerate(zip(data, name)):
        axes[i].set_title(n)
        X = X.data if isinstance(X, csr_matrix) else X.flatten()
        axes[i].hist(X, bins=40)
        if log:
            axes[i].set_yscale('log')


def top_features(matrix_pmi: np.array, k: int = 700) -> List[int]:
    """
    Создаёт множество признаков из выборки (топ k токенов) для каждого класса
    :param matrix_pmi: матрица PMI
    :param k: топ признаков
    """
    features = set()
    for label in range(matrix_pmi.shape[1]):
        token_idxs = np.argsort(-matrix_pmi[:, label].ravel())[:k]
        features = features.union(set(token_idxs))
    return list(features)
