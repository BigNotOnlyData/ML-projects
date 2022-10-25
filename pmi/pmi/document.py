from collections import Counter
from typing import Tuple, List

import numpy as np
from scipy.sparse import csr_matrix


class PmiDocument:
    """
    Класс для расчета PMI документов

    Available Methods:
        fit - обязательный метод для построения матриц PMI
        transform - векторизация документов на основе матриц PMI
        relevant_tokens_for_class - топ релевантных токенов для класса

    Available Attributes:
            tok2indx - словарь токен: индекс
            indx2tok - словарь индекс: токен
            matrix_pmi - матрица PMI
            matrix_lmi - матрица LMI
            matrix_npmi - матрица NPMI
    """
    def __init__(self, min_count=1, max_doc_freq=1.0, ppmi=True) -> None:
        """
        :param min_count: минимальное число документов для токена
        :param max_doc_freq: максимальная доля документов для токена
        :param ppmi: флаг для расчета Positive PMI, иначе рассчитывается обычный PMI.
        """
        self.ppmi = ppmi
        self.MIN_COUNT = min_count
        self.MAX_DOC_FREQ = max_doc_freq
        self.N_DOC = None
        self.N_CLASS = None
        self.N_UNIQUE_TOKENS = None
        self.tok2indx = None
        self.indx2tok = None
        self.matrix_pmi = None
        self.matrix_lmi = None
        self.matrix_npmi = None

    def build_vocabular(self, corpus: List[List[str]]) -> None:
        """
        Создаёт словари уникальных токенов.
        :param corpus: токенизированный текст.
        """
        word_doc_counts = Counter()
        # Встречаемость токенов в документах (без повтора)
        for txt in corpus:
            for token in set(txt):
                word_doc_counts[token] += 1

        # фильтрация токенов
        word_doc_counts = {word: cnt for word, cnt in word_doc_counts.items()
                           if cnt >= self.MIN_COUNT and cnt / len(corpus) <= self.MAX_DOC_FREQ}

        # сортировка по количеству документов
        word_doc_counts = dict(sorted(word_doc_counts.items(), key=lambda x: x[1], reverse=True))

        # словари токенов
        self.tok2indx = {k: i for i, k in enumerate(word_doc_counts.keys())}  
        self.indx2tok = {v: k for k, v in self.tok2indx.items()}
        self.N_UNIQUE_TOKENS = len(self.tok2indx)
    
    def generate_matrix_wc(self, corpus: List[List[str]], labels: np.array) -> np.array:
        """
        Создаёт матрицу частот документов относительно токена и класса
        :param corpus: токенизированный текст.
        :param labels: метки документов
        """
        wc_matrix = np.zeros(shape=(self.N_UNIQUE_TOKENS, self.N_CLASS), dtype=np.int32)

        for text, label in zip(corpus, labels):
            # подсчет документов для токена
            for token in set(text):
                # блок try-except для фильтрации токенов по словарю
                try:
                    self.tok2indx[token]
                except KeyError:
                    continue
                wc_matrix[self.tok2indx[token], label] += 1 
        return wc_matrix        
            
    def generate_martix_pmi(self, wc_matrix: np.array, nc: np.array) -> None:
        """
        Расчитывает матрицы PMI, LMI, NPMI размером (кол-во токенов, кол-во классов)
        :param wc_matrix: матрица частотности документов размером (кол-во токенов, кол-во классов)
        :param nc: массив количества документов для каждого класса
        """
        row, col = wc_matrix.shape
        
        n_doc = nc.sum()

        assert nc.shape == (col,), f"Неверный размер массива классов {nc.shape}"

        p_w = (wc_matrix.sum(1) / n_doc).reshape(-1, 1)  # вероятность слова
        assert p_w.shape == (row, 1), f"Неверный размер массива слов {p_w.shape}"

        p_wc = wc_matrix / nc  # условная вероятность слова в классе
        assert p_wc.shape == (row, col), f"Неверный размер массива условной вероятности {p_wc.shape}"

        p_w_c = wc_matrix / n_doc  # совместная вероятность слово+класс
        assert p_w_c.shape == (row, col), f"Неверный размер массива совместной вероятности {p_w_c.shape}"

        pmi = np.log2(p_wc / p_w)
        
        # Положительный PMI
        if self.ppmi:
            pmi = np.where(pmi > 0, pmi, 0)  # PPMI
        assert pmi.shape == (row, col), f"Неверный размер массива pmi {pmi.shape}"

        lmi = p_w_c * pmi  # локальный PMI
        npmi = pmi / -np.log2(p_w_c)  # нормализованный PMI
        
        self.matrix_pmi = pmi
        self.matrix_lmi = lmi
        self.matrix_npmi = npmi
    
    def doc_vectorize_max(self, corpus: List[List[str]], matrix_pmi: np.array) -> csr_matrix:
        """
        Векторизация документов на основе матрицы PMI. Для каждого токена выбирается
        максимальное значение из PMI, относительно классов.
        return векторизованный корпус текстов
        :param corpus: токенизированный текст.
        :param matrix_pmi: матрица PMI
        :return: векторизованный исходный корпус
        """
        # данные для разряженной матрицы
        row_indxs = []
        col_indxs = []
        dat_values_pmi = []

        for i_doc, text in enumerate(corpus):
            tokens = []  # токены в виде индексов
            for token in set(text):
                # фильтр для отсутсвующих токенов
                try:
                    tokens.append(self.tok2indx[token])
                except KeyError:
                    continue

            data = matrix_pmi[tokens].max(1)  # макимальное PMI относительно классов для каждого токена

            # Фильтрация PMI=0
            mask_nonzero = data != 0
            if not mask_nonzero.sum():
                continue

            # оставляем ненулнвые элементы
            data_nonzero = data[mask_nonzero]
            tokens_nonzero = np.array(tokens)[mask_nonzero].tolist()
            rows = [i_doc] * len(tokens_nonzero)

            # заполняем данные для разряженной матрицы
            row_indxs.extend(rows)
            col_indxs.extend(tokens_nonzero)
            dat_values_pmi.extend(data_nonzero)

        doc_matrix = csr_matrix((dat_values_pmi, (row_indxs, col_indxs)),
                                shape=(len(corpus), self.N_UNIQUE_TOKENS)
                                )
        return doc_matrix
    
    def fit(self, corpus: List[List[str]], labels: np.array) -> None:
        """
        Запуск процесса подсчета PMI.
        :param corpus: токенизированный текст.
        :param labels: метки документов.
        """
        assert len(corpus) == len(labels), "Должна быть одинаковая длина корпуса и меток"
        assert isinstance(labels[0], np.integer), 'У меток должен быть целочисленный тип данных'
        
        # кол-во документов в каждом классе (отсортированный массив)
        _, nc = np.unique(labels, return_counts=True)  
        self.N_CLASS = len(nc)
        
        self.build_vocabular(corpus)
        wc_matrix = self.generate_matrix_wc(corpus, labels)
        self.generate_martix_pmi(wc_matrix, nc)
        
    def transform(self, corpus: List[List[str]], mode: str = 'pmi') -> csr_matrix:
        """
        Преобразует исходный корпус в численное представление на основе матрицы PMI
        :param corpus: токенизированный текст.
        :param mode: указатель для выбора матрицы PMI
        """
        if mode == 'pmi':
            return self.doc_vectorize_max(corpus, self.matrix_pmi)
        elif mode == 'lmi':
            return self.doc_vectorize_max(corpus, self.matrix_lmi)
        elif mode == 'npmi':
            return self.doc_vectorize_max(corpus, self.matrix_npmi) 
        else:
            raise ValueError(f"Валидные значения mode: 'pmi', 'lmi', 'npmi'")
            
    def relevant_tokens_for_class(self, matrix_pmi: np.array, label: int,
                                  topn: int = 10) -> List[Tuple[str, float]]:
        """
        Выбирает из matrix_pmi топ слов из класса label.
        :param matrix_pmi: матрица со значениями PMI.
        :param label: метка класса
        :param topn: количество слов в топе.
        """
        label_score = matrix_pmi[:, label]
        idxs = np.argsort(-label_score)[:topn]
        top_tokens = [(self.indx2tok[idx], label_score[idx]) for idx in idxs]
        return top_tokens
            
#     def pmi_emdedding(self, X, vocabulary, matrix_pmi):
#         embedding = np.zeros(shape=(len(X), matrix_pmi.shape[1]))
#         for i_doc, text in tqdm(enumerate(X), total=len(X)):
#             tokens = []  # токены в виде индексов
#             for token in set(text):
#                 # фильтр для отсутсвующих токенов
#                 try:
#                     tokens.append(vocabulary[token])
#                 except KeyError:
#                     continue
#             if len(tokens):
#                 embedding[i_doc] = matrix_pmi[tokens].mean(0) 
#             else:
#                 continue

#         return embedding       
