import os
import nltk
import json
import string
import logging
import collections
import numpy as np
import pandas as pd
from math import ceil
from datetime import timedelta
# from nltk.corpus import wordnet
from text_mining.definitions import ROOT_DIR
from text_mining.src.utility import CONSTANTS
from pattern3.text.en import lexeme
from nltk.stem.wordnet import wordnet
from stop_words import get_stop_words
from configparser import RawConfigParser
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

author = 'Manu Chauhan'


logger = logging.getLogger(__name__)

config = RawConfigParser()
config_file = 'config/config.properties'
config.read(config_file)


def get_file_path_dict():
    """
    Returns dictionary for file name(key) and path(value) relative to ROOT_DIR of the project.
    :return: dict of file paths
    """

    if not hasattr(get_file_path_dict, "file_path_dict"):
        get_file_path_dict.file_path_dict = dict(config.items('File_Section'))

    return get_file_path_dict.file_path_dict


def get_db_details():
    """
    Reads key value pairs from 'Db_Section' in config.properties file and forms a dictionary for the same
    :return: the dictionary of key value pairs in 'config.properties' file
    """

    if not hasattr(get_db_details, 'db_dict'):
        get_db_details.db_dict = dict(config.items('Db_Section'))

    return get_db_details.db_dict


def get_sql_dict():
    """
    used for retrieving sql from config file where sql reads texts from database
    :return: dict of SQLs
    """

    if not hasattr(get_sql_dict, "sql_dict"):
        get_sql_dict.sql_dict = dict(config.items('Sql_Section'))

    return get_sql_dict.sql_dict


def get_wordnet_pos(tag):
    """
    Converts nltk.pos tag to WordNet part of speech name
    :param tag: nltk.pos tag
    :return: WordNet pos name that WordNetLemmatizer can use
    """

    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return False


def process_data(data, type=None):
    """
    Processes and cleans tha data after doing Part Of Speech tagging.Cleaning involves removing stop words, punctuations
     and numeric or alphanumeric tokens and keeping only Nouns, Verbs, Adverbs and Adjectives.
     WordNetLemmatizer is used for lemmatizing words which takes word and pos tag.
    :param data: list of data
    :param type: type of data processing, possible values : None
        ( used when training the data, filters Noun, Verb, Adverb, Adjective),
        CONSTANTS.NO_ADJ_ADV : does not consider Adjectives and Adverbs
    :return: clean data involving only Nouns, Verbs, Adverbs and Adjectives
    """

    logger.debug('Cleaning data')

    texts = []
    stop_words = set(get_stop_words('en'))
    punctuations = set(string.punctuation)
    lemmatizer = WordNetLemmatizer()

    # [stop_words.discard(i) for i in set(CONSTANTS.TO_KEEP)]

    for text in data:
        tags = nltk.pos_tag(nltk.word_tokenize(text.lower()))

        clean_txt = [(word, get_wordnet_pos(pos)) for word, pos in tags if word not in stop_words and
                     word not in punctuations and len(word) >= 3 and not word.isdigit() and
                     word.isalpha() and get_wordnet_pos(pos) is not False] if type is None else \
            [(word, get_wordnet_pos(pos)) for word, pos in tags if word not in stop_words and
             not pos.startswith('J') and not pos.startswith('R') and word not in punctuations and len(
                word) >= 3 and not word.isdigit() and
             word.isalpha() and get_wordnet_pos(pos) is not False] if type == CONSTANTS.NO_ADJ_ADV else []

        if len(clean_txt) > 0:
            temp = [lemmatizer.lemmatize(word=word, pos=pos) for word, pos in clean_txt]
        else:
            temp = ['']
        texts.append(' '.join(temp))

    logger.debug('Cleaning data done')

    return texts


def get_train_data_from_file():
    """
    Retrieves train data from 'texts.csv' file under train_data directory.
    :return: texts and labels as lists
    """

    df = pd.read_csv('train_data/texts.csv', header=None)

    texts, labels = [], []

    for row in df.itertuples(index=False):
        txt, lbl = row
        if lbl in ['p', 'n']:
            # lbl = 1 if lbl == 'p' else 0
            texts.append(txt)
            labels.append(lbl)

    logger.info(' Texts found in file for get_train_data_file has size = ', len(texts))

    # texts = df.iloc[:, 0:1].values
    # labels = df.iloc[:, 1:2].values

    return texts, labels


def get_last_date_of_month(date):
    if date.month == 12:
        return date.replace(day=31)
    return date.replace(month=date.month + 1, day=1) - timedelta(days=1)


def get_predicted_percentage(data):
    positive_count = sum(1 for item in data if item.lower() == 'p')
    pos_perc = (positive_count / len(data)) * 100
    neg_perc = 100 - pos_perc
    return round(pos_perc, 2), round(neg_perc, 2)


def get_week_of_month(date):
    """
    Returns the week of the month for the date argument passed considering Sunday as week start.

    :param date: date whose week_num is to be determined
    :return: week number of the month (int)
    """

    first_day = date.replace(day=1)

    dom = date.day
    adjusted_dom = dom + first_day.weekday() + 1

    return int(ceil(adjusted_dom / 7.0))


def get_formatted_month_year_name(date):
    """
    Formats the month year name for the :param date
    :param date: date object
    :return: string for formatted month and year name eg: 'Jan-2017'
    """
    return date.strftime("%b") + '-' + date.strftime("%Y")


def get_word_cloud_data(data, sentiment):
    """
    Calculates average score of each feature in the TFIDFed conversion on the data passed and
     returns the json of 'word':score elements.
    Uses the :param sentiment for choosing 'min_df' and 'max_df' for TfIdf vectorizer as
    the number of features vary as per sentiment in the data.

    :param data: data from which to form the word cloud weights
    :param sentiment: used to decide 'min_df' and 'max_df' for tfidf vectorizer
    :return: 1. list of words (dict keys) 2. json for word:weight elements
    """

    min_df, max_df = (0.005, 0.95) if sentiment == CONSTANTS.POSITIVE else (0.015, 0.95)
    tfidf_vect = TfidfVectorizer(min_df=min_df, max_df=max_df)
    tfidfed = tfidf_vect.fit_transform(data).toarray()

    new_vocab = {y: x for x, y in tfidf_vect.vocabulary_.items()}
    vocab_list = sorted([(x, y) for x, y in new_vocab.items()], key=lambda x: x[0])

    ''' removed loop mechanism and added np.mean along axis = 0, faster than normal loop
    calculating score for each feature. Now calculates mean in place of highest score for each feature.
    '''

    scores = np.mean(tfidfed, axis=0).tolist()

    # for c in range(0, tfidfed.shape[1]):
    #     highest_score = 0
    #     for r in range(0, tfidfed.shape[0]):
    #         if tfidfed[r][c] >= highest_score:
    #             highest_score = tfidfed[r][c]
    #
    #     scores.append(highest_score)

    indexes, words = zip(*vocab_list)

    new_list = sorted(list(zip(indexes, words, scores)), key=lambda x: x[0])

    word_dict = collections.defaultdict(float)

    for index, word, weight in new_list:
        if wordnet.synsets(word):
            word_dict[word] = weight

    # word_cloud_list = []
    #
    # for index, word, weight in new_list:
    #     if wordnet.synsets(word):
    #         word_weight = {'word': word, 'weight': weight}
    #         word_cloud_list.append(word_weight)

    return list(word_dict.keys()), word_dict


def separate_single_sentiment_data(data, sentiment):
    """
    Separates out data for type :param sentiment from :param data and returns as list of single :param sentiment
    :param data: data containing more than 1 sentiment
    :param sentiment: the sentiment for which to filter the data
    :return: list of texts of one type of :param sentiment
    """
    return [text for text, predicted in data if predicted.lower() == sentiment]


def get_verbatims_for_word(data, words, sentiment, last_date, month_year_name, agent_id):
    """
    Finds corresponding texts for each word in :param words from word cloud. And forms list of tuples where
     each tuple is like : (last_date, word, verbatim list, agent_nt_id, month_year, sentiment) using corresponding params

    :param data: list of texts of type only :param sentiment
    :param words: list words used for word cloud ( word cloud dict keys )
    :param sentiment: sentiment type for which :param data is passed.
    :param last_date: last date of month
    :param month_year_name: month year formatted string
    :param agent_id: agent id for whom :param data is processed
    :return: list of tuples which could be directly passed for inserting into database
    """

    return_data = []

    for word in words:
        verbatim_list = []
        for verbatim in data:
            # if re.compile(r'\b({0})\b'.format(word), flags=re.IGNORECASE).search(verbatim) and len(verbatim) > 0:

            words_set = set(verbatim.lower().split(' '))

            if any(item in words_set for item in lexeme(word)):
                verbatim_list.append(verbatim)
                # for item in words_set:
                #     if item in lexeme(word):
                #         # if item.startswith(word.lower()) or item.startswith(word.lower()[:-1]):
                #         verbatim_list.append(verbatim)
                #         break

        return_data.append((last_date, word, verbatim_list[:1000], agent_id, month_year_name, sentiment))

    return return_data


def prepare_word_cloud_data(last_date, verbatims_for_words, word_cloud_dict, sentiment, nt_id, month_year):
    words_to_remove = set(item[1] for item in verbatims_for_words if len(item[2]) == 0)

    new_word_cloud_dict = {x: y for x, y in word_cloud_dict.items() if
                           x not in words_to_remove}

    word_cloud_json = json.dumps(new_word_cloud_dict)

    word_cloud_data = [
        (last_date, word_cloud_json, sentiment, nt_id, month_year)]

    return word_cloud_data
