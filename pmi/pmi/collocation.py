from collections import Counter
from typing import Tuple, List

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class PmiCollocation:
    """
    Класс для расчета PMI словосочетаний

    Available Methods:
        fit - обязательный метод для построения матриц PMI
        global_best_pmi - топ по всему корпусу
        local_best_pmi - топ по переданному слову
        word_cos_similarity - похожие слова к переданному слову

    Available Attributes:
            tok2indx - словарь токен: индекс
            indx2tok - словарь индекс: токен
            matrix_pmi - матрица PMI
            matrix_lmi - матрица LMI
            matrix_npmi - матрица NPMI
    """

    def __init__(self, min_words: int = 1, window: int = 2, ppmi: bool = True) -> None:
        """
        :param min_words: минимальная частота для токена в корпусе.
        :param window: радиус для поиска контекстуальных токенов.
        :param ppmi: флаг для расчета Positive PMI, иначе рассчитывается обычный PMI.
        """
        self.min_words = min_words
        self.window = window
        self.ppmi = ppmi
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
        unigram_counts = Counter()
        for text in tqdm(corpus, desc='Vocabular'):
            for token in text:
                unigram_counts[token] += 1

        # фильтрация по количеству токенов
        unigram_counts = {k: cnt for k, cnt in unigram_counts.items() if cnt >= self.min_words}

        # сортировка по количеству токенов
        unigram_counts = dict(sorted(unigram_counts.items(), key=lambda x: x[1], reverse=True))

        # словари токенов
        self.tok2indx = {tok: indx for indx, tok in enumerate(unigram_counts.keys())}
        self.indx2tok = {indx: tok for tok, indx in self.tok2indx.items()}

    def generate_skipgrams(self, corpus: List[List[str]]) -> Counter:
        """
        Подсчитывает количество встречаемости в корпусе каждой пары токенов
        в радиусе self.window впереди и сзади каждого токена.
        :param corpus: токенизированные тексты.
        :return: словарь, где ключ - пара токенов, значение - частота
        """
        blank = -1  # маска для отсутсвующего в словаре слова
        skipgram_counts = Counter()
        for text in tqdm(corpus, desc='Skipgrams'):
            # формируем массив индексов токенов 
            tokens = []
            for tok in text:
                try:
                    tokens.append(self.tok2indx[tok])
                except KeyError:
                    tokens.append(blank)

            # Выдеялем контекстные слова вокруг центрального
            for i_word, word in enumerate(tokens):
                if word == blank:
                    continue

                i_context_min = max(0, i_word - self.window)
                i_context_max = min(len(tokens) - 1, i_word + self.window)
                i_contexts = [ii for ii in range(i_context_min, i_context_max + 1) if ii != i_word]
                for i_context in i_contexts:
                    # фильтрация одинаковых токенов в паре и пропущенных токенов
                    if word != tokens[i_context] and tokens[i_context] != blank:
                        skipgram = (word, tokens[i_context])
                        skipgram_counts[skipgram] += 1
        return skipgram_counts

    def generate_count_matrix(self, skipgram_counts: Counter) -> coo_matrix:
        """
        Создаёт квадратную разряженную матрицу (N, N) частоты совстречаемости слов из словаря,
        где N - количество уникальных токенов.
        :param skipgram_counts: словарь частот пары токенов
        """
        row_indxs = []
        col_indxs = []
        dat_values = []
        for (tok1, tok2), sg_count in tqdm(skipgram_counts.items(), desc='Matrix'):
            row_indxs.append(tok1)
            col_indxs.append(tok2)
            dat_values.append(sg_count)

        ww_matrix = coo_matrix((dat_values, (row_indxs, col_indxs)))
        return ww_matrix

    def generate_matrix_pmi(self, ww_matrix: coo_matrix) -> None:
        """
        Расчитывает квадратные матрицы PMI, LMI, NPMI размером (N, N), где N - кол-во уникальных токенов.
        :param ww_matrix: матрица частотности совместной встречаемости токенов размером (N, N)
        """

        ww_matrix = ww_matrix.tocsr()
        row, col = ww_matrix.shape
        n_total = ww_matrix.sum()  # всего пар слов

        nw2 = ww_matrix.sum(0)  # массив количества пар слов в контексте
        assert nw2.shape == (1, col), f"Неверный размер массива слов {nw2.shape}"

        nw1 = ww_matrix.sum(1)  # массив количества пар слов с центральным словом
        assert nw1.shape == (row, 1), f"Неверный размер массива слов {nw1.shape}"

        p_w1 = nw1 / n_total  # вероятность центрального слова w1
        assert p_w1.shape == (row, 1), f"Неверный размер массива p_w1 {p_w1.shape}"

        p_w1w2 = ww_matrix.multiply(1 / nw2)  # условная вероятность центрального слова w1 в контекcте слова w2
        assert p_w1w2.shape == (row, col), f"Неверный размер массива условной вероятности {p_w1w2.shape}"

        p_w1_w2 = ww_matrix.multiply(1 / n_total)  # совместная вероятность слов w1 и w2
        assert p_w1_w2.shape == (row, col), f"Неверный размер массива совместной вероятности {p_w1_w2.shape}"

        # у разряженной матрицы нет метода log2, поэтому используем .data
        pmi = p_w1w2.multiply(1 / p_w1)
        pmi_log = np.log2(pmi.data)

        # PPMI
        if self.ppmi:
            pmi.data = np.where(pmi_log > 0, pmi_log, 0)
        else:
            pmi.data = pmi_log

        assert pmi.shape == (row, col), f"Неверный размер массива pmi {pmi.shape}"

        # LMI и NPMI 
        lmi = p_w1_w2.multiply(pmi)  # локальный PMI
        npmi = pmi.copy()
        npmi.data = pmi.data / -np.log2(p_w1_w2.data)  # нормализованный PMI

        # исключаем возможные нули из данных
        pmi.eliminate_zeros()
        lmi.eliminate_zeros()
        npmi.eliminate_zeros()

        # Трансформируем матрицы в формат CSR для работы со строками
        self.matrix_pmi = pmi.tocsr()
        self.matrix_lmi = lmi.tocsr()
        self.matrix_npmi = npmi.tocsr()

    def global_best_pmi(self, sparse_matrix: csr_matrix, topn: int = 10) -> List[Tuple[str, str, float]]:
        """
        Выбирает из всей sparse_matrix топ пар слов.
        :param sparse_matrix: матрица со значениями PMI.
        :param topn: количество слов в топе.
        """
        matrix = sparse_matrix.tocoo()
        index = np.argsort(-matrix.data)[:topn]
        best_score = [(self.indx2tok[row], self.indx2tok[col], score) for row, col, score
                      in zip(matrix.row[index], matrix.col[index], matrix.data[index])]
        return best_score

    def local_best_pmi(self, sparse_matrix: csr_matrix, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Выбирает из sparse_matrix топ слов из контекста word.
        :param sparse_matrix: матрица со значениями PMI.
        :param word: центральное слово.
        :param topn: количество слов в топе.
        """
        index = self.tok2indx[word]
        scores = sparse_matrix.getrow(index).A.flatten()
        best_index = np.argsort(-scores)[:topn]
        best_context = [(self.indx2tok[ind], scores[ind]) for ind in best_index]
        return best_context

    def word_cos_similarity(self, word, mat: csr_matrix, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Вычисляет топ наиболее похожих слов к переданному слову
        :param word: главное слово.
        :param mat: матрица со значениями PMI.
        :param topn: количество слов в топе.
        """
        indx = self.tok2indx[word]
        v1 = mat.getrow(indx)
        sims = cosine_similarity(mat, v1).flatten()
        sindxs = np.argsort(-sims)[:topn]
        sim_word_scores = [(self.indx2tok[sindx], sims[sindx]) for sindx in sindxs]
        return sim_word_scores

    def fit(self, corpus: List[List[str]]) -> None:
        """
        Запуск процесса подсчета PMI
        :param corpus: токенизированные тексты.
        """
        self.build_vocabular(corpus)
        skipgrams = self.generate_skipgrams(corpus)
        count_matrix = self.generate_count_matrix(skipgrams)
        self.generate_matrix_pmi(count_matrix)
