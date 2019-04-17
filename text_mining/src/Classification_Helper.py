import os
import pickle
import logging
from text_mining.src.utility import MyUtils
from text_mining.definitions import ROOT_DIR
from text_mining.src.utility import CONSTANTS
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

author = 'Manu Chauhan'

logger = logging.getLogger(__name__)


def split_train_test_data(features, targets, size=0.20):
    """
    Splits data into separate train and test sets with random_state = 1 (random_state can be changed)

    :param features: features or texts
    :param targets: targets or labels
    :param size: test size, if float then should be between 0.0 or 1.0 and float value is that proportion of total data size,
                 if int then exactly that many number of samples as test size.
                 If size is not provided then default value is set as 0.20 in this function

    :return: X_train, X_test, y_train, y_test
    """

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=size, random_state=1)

    return x_train, x_test, y_train, y_test


def get_count_vectorizer():
    if not hasattr(get_count_vectorizer, 'count_vectorizer'):
        get_count_vectorizer.count_vectorizer = CountVectorizer(stop_words='english')
    return get_count_vectorizer.count_vectorizer


def get_tfidf_vectorizer():
    if not hasattr(get_tfidf_vectorizer, 'tfidf_vect'):
        get_tfidf_vectorizer.tfidf_vect = TfidfVectorizer(stop_words='english')
    return get_tfidf_vectorizer.tfidf_vect


def get_tfidf_transformer():
    if not hasattr(get_tfidf_transformer, 'tfidf_trans'):
        get_tfidf_transformer.tfidf_trans = TfidfTransformer()
    return get_tfidf_transformer.tfidf_trans


def get_classifier(classifier=MultinomialNB()):
    if not hasattr(get_classifier, 'clf'):
        get_classifier.clf = classifier
    return get_classifier.clf


def train_model(X_data, y_data):
    """
    Trains the model by using the parameter values. Default classifier is MultinomialNB
    Also stores the trained model and fitted tfidf vectorizer on disk with file names 'trained_model' and 'tfidf_vect' under utility dir
    :return: Fitted model and fitted tfidf_vectorizer
    """

    file_path_dict = MyUtils.get_file_path_dict()

    tfidf_vect = get_tfidf_vectorizer()
    classifier = get_classifier()
    X_tfidfed = tfidf_vect.fit_transform(X_data)
    classifier.fit(X_tfidfed, y_data)

    with open(file_path_dict[CONSTANTS.TRAINED_MODEL], 'wb') as file:
        pickle.dump(classifier, file)
    with open(file_path_dict[CONSTANTS.TFIDF_VECTORIZER], 'wb') as file:
        pickle.dump(tfidf_vect, file)

    return classifier, tfidf_vect


def classify_new_data(x_new):
    """
    Loads the pickled classifier and tfidf vectorizer then uses those for transforming and classifying new data
    :param x_new: New data that needs to be classified
    :return: prediction on new data
    """

    file_path_dict = MyUtils.get_file_path_dict()

    with open(file_path_dict[CONSTANTS.TRAINED_MODEL], 'rb') as file:
        classifier = pickle.load(file)
    with open(file_path_dict[CONSTANTS.TFIDF_VECTORIZER], 'rb') as file:
        tfidf_vect = pickle.load(file)

    predicted = classifier.predict(tfidf_vect.transform(x_new))
    return predicted