import re
import string
from typing import List
from pymystem3 import Mystem


def lowercase(text: str) -> str:
    """
    Приводит все символы в нижний регистр
    """
    return text.lower()

def remove_emojis(text: str) -> str:
    """
    Удаляет смайлы по коду юникода
    """
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, ' ', text)

def remove_html(text: str) -> str:
    """
    Удаление html кода
    """
    pattern = re.compile('<.*?>')
    return pattern.sub(' ', text)

def remove_urls(text: str) -> str:
    """
    Удаление некоторых ссылок
    """
    re_urls = ['https?://\S+', 'www\.\S+', 'bit.ly/\S+']
    url_pattern = re.compile(r'|'.join(re_urls))
    return url_pattern.sub(' ', text)

def remove_spec_symbols(text: str) -> str:
    """
    Удаление спецсимволов
    """
    pattern = re.compile(r'&\S+?;')
    return pattern.sub(' ', text)

def remove_files(text: str) -> str:
    """
    Удаление некоторых файлов
    """
    pattern = re.compile(r'\S+\.(?:csv|xlsx|py|ipynb|pdf|zip|rar|docx)')
    return pattern.sub(' ', text)

def remove_punctuation(text: str) -> str:
    """
    Удаление знаков пунктуации
    """
    pattern = re.compile(fr'[{string.punctuation}]')
    return pattern.sub(' ', text)

def remove_digits(text: str) -> str:
    """
    Удаление цифр
    """
    pattern = re.compile(r'\d+')
    return pattern.sub(' ', text)

def only_ru_en_chars(text: str) -> str:
    """
    Дополнительная очистка от небуквенных символов.
    Казалось бы почему сразу не применить эту функцию, 
    но тогда останутся внутренности html, ссылок и подобные вещи
    """
    pattern = re.compile(r'[^a-zа-яё ]')
    return pattern.sub(' ', text)

def remove_all_non_russian(text: str) -> str:
    """
    Обнуляет строки, состоящие полностью на не русском языке
    """
    pattern = re.compile(r'[а-яё]')
    return text if re.search(pattern, text) else ''

def remove_whitespace(text: str) -> str:
    """
    Удаляет лишние пробелы 
    """
    pattern = re.compile(r'\s+')
    return pattern.sub(' ', text)

def tokenizer(text: str) -> List[str]:
    """
    Токенизация теста
    :param text: текст
    :return: список токенов
    """
    return text.split()

def lemmatization(list_text: List[str], stem: Mystem) -> List[str]:
    """
    Лемматизирует слова
    :param list_text: список текстов
    :param stem: стеммер pymystem3
    :return: список текстов со словами в нормальной форме
    """
    # соединяем все тексты в один большой
    text = " # ".join(list_text)
    # лемматизируем с pymystem
    lemmas = stem.lemmatize(text)
    # возвращаем текст в исходный вид
    return ''.join(lemmas).split(' # ')

def remove_stopwords(token_list: List[str], stopwords: set) -> List[str]:
    """
    Удаляет стоп-слова из списка токенов.
    :param token_list: список токенов
    :param stopwords: множество стоп слов
    :return: список токенов без стоп-слов
    """
    return [word for word in token_list if word not in stopwords]

def text_filter(tokenized_text: List[List[str]], min_symbols: int, min_words: int) -> List[List[str]]:
    """
    Фильтрует тексты по минимальному количеству токенов и символов и удаляет дубликаты.
    :param tokenized_text: список токенизированных текстов
    :param min_symbols: минимум символов для токенов
    :param min_words: минимум слов для теста
    :return: писок токенизированных, отфильтрованных текстов
    """
    # фильтрация по минимальному количеству символов (букв)
    filtered_tokenized_text = [[word for word in text if len(word) >= min_symbols]
                               for text in tokenized_text]

    # фильтрация по минимальному количеству слов в комментарие
    filtered_tokenized_text = [text for text in filtered_tokenized_text if len(text) >= min_words]

    # Удаление дубликатов строк
    filtered_tokenized_text = set([' '.join(text) for text in filtered_tokenized_text])
    filtered_tokenized_text = list(map(str.split, filtered_tokenized_text))
    return filtered_tokenized_text

def preprocessing(text: str) -> str:
    """
    Выполняеет основную предобработку строк
    """
    text = lowercase(text)
    text = remove_emojis(text)
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_spec_symbols(text)
    text = remove_files(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = only_ru_en_chars(text)
    text = remove_all_non_russian(text)
    text = remove_whitespace(text)
    return text

def get_clean_text(text_corpus: List[str], 
                   stopwords: set, 
                   stem: Mystem, 
                   min_symbols: int, 
                   min_words: int) -> List[List[str]]:
    """
    Главная функция для очистки и предобработки текста
    :param text_corpus: список сырых текстов
    :param stopwords: множество стоп слов
    :param stem: стеммер pymystem3
    :param min_symbols: минимум символов для токенов
    :param min_words: минимум слов для теста
    :return: список токенизированных, очищенных текстов
    """
    # Основная предобработка тектса
    preproces_text = [preprocessing(text) for text in text_corpus]
    # лемматизируем
    lemma_text = lemmatization(preproces_text, stem)
    # токенизируем
    tokenized_text = [tokenizer(text) for text in lemma_text]
    # удаляем стоп слова
    tokenized_text = [remove_stopwords(tokens, stopwords) for tokens in tokenized_text]
    # фильтруем по количеству символов\слов\дубликаты
    clean_text = text_filter(tokenized_text, min_symbols, min_words)
    return clean_text
