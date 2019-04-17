import os
import sys
import logging
import traceback
import pandas as pd
import psycopg2 as pg
from datetime import datetime
# from langdetect import detect
from text_mining.src.utility import MyUtils
from text_mining.definitions import ROOT_DIR
from text_mining.src.utility import CONSTANTS
# from guess_language.guess_language import guessLanguage
from guess_language import guessLanguage

logger = logging.getLogger(__name__)


def get_train_data_from_db():
    """
    Retrieves train data from database based on the sql query
     provided in the config file where sql name is 'get_train_data'
    :return: data(texts) and label(targets)
    """

    print(' Retrieving train data from db ')

    conn = None
    db_details = MyUtils.get_db_details()
    sql = MyUtils.get_sql_dict()['get_training_data']

    try:
        conn = pg.connect(database=db_details['name'], user=db_details['user'],
                          password=db_details['password'], host=db_details['host'], port=db_details['port'])

        logger.debug(' Read query = " %s "', sql)

        data_frame = pd.read_sql(sql, conn)
        data = data_frame.text
        label = data_frame.label

    except (pg.Error, Exception):
        logger.error('Exception in get_train_data in Classification_Helper.py: %s ', traceback.format_exc())
        sys.exit()

    else:
        print('Success in get_train_data()')
        return data, label

    finally:
        if conn:conn.close()


def get_new_data(date):
    """
    Fetches new data from database on which prediction needs to be performed
    :return: list of tuples of data fetched from database
    """

    print(' Retrieving new data from db ')

    conn = None
    db_details = MyUtils.get_db_details()
    sql = MyUtils.get_sql_dict()['get_new_data']

    try:
        conn = pg.connect(database=db_details['name'], user=db_details['user'],
                          password=db_details['password'], host=db_details['host'], port=db_details['port'])

        sql = sql.format(date.strftime('%Y-%m-%d'))
        logger.debug(' Read query = " %s "', sql)

        data_frame = pd.read_sql(sql, conn)

        if len(data_frame.index) == 0:
            print('Data frame empty. No data after date : ', date)
            raise Exception

    except (pg.Error, Exception):
        logger.error('Exception in get_new_data in Classification_Helper.py: %s ', traceback.format_exc())
        sys.exit()
    else:
        data = []
        for row in data_frame.itertuples():
            data.append((row.text, row.nt_id))

        print('Success in get_new_data()')
        return data
    finally:
        if conn:conn.close()


def get_new_data_date_range(date1, date2):
    """
    Fetches new data from database for given date range
    :return: list of tuples of data fetched from database
    """

    logger.info(' Retrieving new data from db ')

    conn = None
    db_details = MyUtils.get_db_details()
    sql = MyUtils.get_sql_dict()['get_new_data_date_range']
    # last_date = get_last_updated_date()

    try:
        conn = pg.connect(database=db_details['name'], user=db_details['user'],
                          password=db_details['password'], host=db_details['host'], port=db_details['port'])

        date1 = date1.strftime('%Y-%m-%d') if not isinstance(date1, str) else date1
        date2 = date2.strftime('%Y-%m-%d') if not isinstance(date2, str) else date2

        sql = sql.format(date1, date2)

        logger.debug(' Read query = " %s "', sql)

        data_frame = pd.read_sql(sql, conn)

        if len(data_frame.index) == 0:
            logger.debug('\n Data frame empty for %s to %s' % (date1, date2))

    except (pg.Error, Exception):
        logger.error('Exception in get_new_data_date_range in Dao.py : %s', traceback.format_exc())
        sys.exit()

    else:
        data = []
        for row in data_frame.itertuples():
            text = row.text.strip()
            if len(text) > 0 and guessLanguage(text) in ['en', 'UNKNOWN']:
                level_1 = "null" if row.level_1 is None else row.level_1
                level_3 = "null" if row.level_3 is None else row.level_3
                date = row.date.strftime("%Y-%m-%d")
                data.append((text, level_1, level_3, row.attuid, date, row.sid))

        logger.info('Success in get_new_data_date_range()')
        return data

    finally:
        if conn:conn.close()


def get_last_updated_date(type):
    """
    Retrieves last entry date for predicted or raw data in database depending upon type.

    :param type: one of 'PREVIOUS' or 'NEW' and is used to decide whether to return last(max) predicted data date or
     max new data date

    :return: date object
    """

    conn = None
    cursor = None
    db_details = MyUtils.get_db_details()
    sql = MyUtils.get_sql_dict()['get_last_result_date'] if type == CONSTANTS.PREVIOUS else MyUtils.get_sql_dict()[
        'get_max_new_data_date'] if type == CONSTANTS.NEW else None

    try:
        conn = pg.connect(database=db_details['name'], user=db_details['user'],
                          password=db_details['password'], host=db_details['host'], port=db_details['port'])

        cursor = conn.cursor()

        logger.debug(' Read query = " %s "', sql)

        cursor.execute(sql)
        date = cursor.fetchone()

        return date[0] if date[0] is not None else datetime(year=datetime.now().year, month=1, day=1).date()

    except Exception:
        logger.error("Exception in get_last_date : %s ", traceback.format_exc())
        sys.exit()

    finally:
        if cursor:cursor.close()
        if conn:conn.close()


def get_max_db_date():
    conn = None
    cursor = None
    db_details = MyUtils.get_db_details()
    sql = MyUtils.get_sql_dict()['get_max_db_date']

    try:
        conn = pg.connect(database=db_details['name'], user=db_details['user'],
                          password=db_details['password'], host=db_details['host'], port=db_details['port'])

        cursor = conn.cursor()

        logger.debug(' Read query = " %s "', sql)

        cursor.execute(sql)
        date = cursor.fetchone()

        return date[0] if date[0] is not None else datetime(datetime.now().year, 1, 1)

    except Exception:
        logger.error("Exception in get_last_date : %s ", traceback.format_exc())
        sys.exit()

    finally:
        if cursor:cursor.close()
        if conn:conn.close()


def insert_monthly_words_verbatims(words_verbatims_data):
    """Inserts word and list of verbatim for a specific sentiment for a month into the database
       Uses psycopg2 to connect to db.
       The query used is specified in config.properties file and is named 'insert_monthly_words_verbatims'
       """

    logger.debug(' Inserting monthly_words_verbatims data into database ')

    conn = None
    cursor = None

    db = MyUtils.get_db_details()
    insert_sql = MyUtils.get_sql_dict()['insert_monthly_words_verbatims']

    try:
        conn = pg.connect(database=db['name'], user=db['user'], password=db['password'], host=db['host'],
                          port=db['port'])

        cursor = conn.cursor()

        record_template = ','.join(['%s'] * len(words_verbatims_data))
        logger.debug(' Insert sql = " %s "', insert_sql)
        insert_query = insert_sql.format(record_template)

        logger.debug(' Executing query for insert_word_cloud_data into db ')
        cursor.execute(insert_query, words_verbatims_data)

        conn.commit()
        logger.info(' commit done ')

    except (pg.Error, Exception):
        logger.error(' Exception in insert_word_cloud_data ', traceback.format_exc())
        raise Exception

    else:
        logger.info('Successfully inserted insert_word_cloud_data')

    finally:
        if cursor:cursor.close()
        if conn:conn.close()


def insert_predicted_into_db(data, period_type):
    """
    Inserts predicted data into the database with corresponding label with list of texts and
     issue level_1 and level_3 values as well.
    Uses psycopg2 to connect to db.
     The query used is specified in config.properties file and is named 'insert_predicted_data'
    """

    logger.debug(' Inserting predicted data into database ')

    conn = None
    cursor = None

    db = MyUtils.get_db_details()
    insert_sql = MyUtils.get_sql_dict()['insert_monthly_pos_neg'] if period_type.lower() == CONSTANTS.MONTHLY \
        else MyUtils.get_sql_dict()['insert_weekly_pos_neg'] if period_type.lower() == CONSTANTS.WEEKLY \
        else MyUtils.get_sql_dict()['insert_daily_pos_neg']

    try:
        conn = pg.connect(database=db['name'], user=db['user'], password=db['password'], host=db['host'],
                          port=db['port'])

        cursor = conn.cursor()

        record_template = ','.join(['%s'] * len(data))
        insert_query = insert_sql.format(record_template)
        logger.debug(' Insert sql = " %s "', insert_query)

        logger.debug(' Executing query for inserting data into db ')
        cursor.execute(insert_query, data)

        conn.commit()
        logger.info(' commit done ')

    except (pg.Error, Exception):
        logger.error(' Exception in insert_predicted_into_db: %s ', traceback.format_exc())
        sys.exit()

    else:
        logger.info('Successfully inserted predicted data')

    finally:
        if cursor:cursor.close()
        if conn:conn.close()


def insert_train_data_into_db_from_file():
    """
    Checks if training data table is empty. If yes then reads labelled texts from files and inserts
     into the database with corresponding label. Otherwise does nothing.
     Uses CSV module to read csv files. psycopg2 to connect to db. pandas data frame to hold data read from csv.
     The query used is specified in config.properties file and is named 'insert_training_data'
    """

    if os.path.isfile('train_data/texts.csv'):
        print(' Reading train set from texts.csv file in train_data dir ')

        df = pd.read_csv('train_data/texts.csv', header=None)

        texts = df.iloc[:, 0:1]
        labels = df.iloc[:, 1:2]
        data = list(zip(texts, labels))

        conn = None
        cursor = None

        db = MyUtils.get_db_details()
        insert_sql = MyUtils.get_sql_dict()['insert_training_data']

        try:
            conn = pg.connect(database=db['name'], user=db['user'], password=db['password'], host=db['host'],
                              port=db['port'])

            cursor = conn.cursor()

            record_template = ','.join(['%s'] * len(data))
            insert_query = insert_sql.format(record_template)
            logger.debug(' Insert sql = " %s "', insert_query)

            logger.info(' Executing query for inserting train data into db ')
            cursor.execute(insert_query, data)

            conn.commit()
            print(' commit done ')

        except (pg.Error, Exception):
            logger.error(' Exception in insert_train_data_into_db_from_file: %s ', traceback.format_exc())
            sys.exit()

        else:
            print('Successfully inserted train data')

        finally:
            if cursor:cursor.close()
            if conn:conn.close()


def insert_word_cloud_data(data):
    """Inserts word cloud data into the database words and their weights
    Uses psycopg2 to connect to db.
    The query used is specified in config.properties file and is named 'insert_monthly_sentiment'
    """

    logger.debug(' Inserting word_cloud_data data into database ')

    conn = None
    cursor = None

    db = MyUtils.get_db_details()
    insert_sql = MyUtils.get_sql_dict()['insert_monthly_word_cloud']

    try:
        conn = pg.connect(database=db['name'], user=db['user'], password=db['password'], host=db['host'],
                          port=db['port'])

        cursor = conn.cursor()

        record_template = ','.join(['%s'] * len(data))
        logger.debug(' Insert sql = " %s "', insert_sql)
        insert_query = insert_sql.format(record_template)

        logger.debug(' Executing query for insert_word_cloud_data into db ')
        cursor.execute(insert_query, data)

        conn.commit()
        logger.info(' commit done ')

    except (pg.Error, Exception):
        logger.error(' Exception in insert_word_cloud_data ', traceback.format_exc())
        raise Exception

    else:
        logger.info('Successfully inserted insert_word_cloud_data')

    finally:
        if cursor:cursor.close()
        if conn:conn.close()


def get_min_data_date():
    """
    Gives minimum date from raw data
    :return: date object
    """

    conn = None
    cursor = None
    db_details = MyUtils.get_db_details()
    sql = MyUtils.get_sql_dict()['get_min_db_date']

    try:
        conn = pg.connect(database=db_details['name'], user=db_details['user'],
                          password=db_details['password'], host=db_details['host'], port=db_details['port'])

        cursor = conn.cursor()

        logger.debug(' Read query = " %s "', sql)

        cursor.execute(sql)
        date = cursor.fetchone()

        return date[0] if date[0] is not None else datetime(datetime.now().year, 1, 1).date()

    except Exception:
        logger.error("Exception in get_min_data_date : ", traceback.format_exc())
        raise Exception

    finally:
        if cursor:cursor.close()
        if conn:conn.close()


def insert_into_monthly_sentiment(data):
    """
    Inserts data into the database text, label, issue1, issue3, agent_id
    Uses psycopg2 to connect to db.
    The query used is specified in config.properties file and is named 'insert_monthly_sentiment'
    """

    logger.debug(' Inserting insert_monthly_sentiment data into database ')

    conn = None
    cursor = None

    db = MyUtils.get_db_details()
    insert_sql = MyUtils.get_sql_dict()['insert_monthly_sentiment']

    try:
        conn = pg.connect(database=db['name'], user=db['user'], password=db['password'], host=db['host'],
                          port=db['port'])

        cursor = conn.cursor()

        record_template = ','.join(['%s'] * len(data))
        logger.debug(' Insert sql = " %s "', insert_sql)
        insert_query = insert_sql.format(record_template)

        logger.debug(' Executing query for inserting data into db ')
        cursor.execute(insert_query, data)

        conn.commit()
        logger.info(' commit done ')

    except (pg.Error, Exception):
        logger.error(' Exception in insert_into_monthly_sentiment', traceback.format_exc())
        raise Exception

    else:
        logger.info('Successfully inserted predicted data')

    finally:
        if cursor:cursor.close()
        if conn:conn.close()


def delete_previous_data(tables, period):
    """
    Deletes previous data from database for table in 'tables' list
    :return: None
    """
    sql = None
    conn = None
    cursor = None
    db_details = MyUtils.get_db_details()

    try:

        for table in tables:
            if period.lower() == CONSTANTS.ALL:
                sql = MyUtils.get_sql_dict()[
                    'delete_monthly_sentiment_data'] if table.lower() == CONSTANTS.MONTHLY_SENTIMENT \
                    else MyUtils.get_sql_dict()[
                    'delete_monthly_word_cloud'] if table.lower() == CONSTANTS.MONTHLY_WORD_CLOUD else \
                    MyUtils.get_sql_dict()[
                        'delete_monthly_words_verbatims'] if table.lower() == CONSTANTS.MONTHLY_WORDS_VERBATIMS else None

            elif period.lower() == CONSTANTS.PREVIOUS:
                sql = MyUtils.get_sql_dict()[
                    'delete_monthly_sentiment_data_last_month'] if table.lower() == CONSTANTS.MONTHLY_SENTIMENT \
                    else MyUtils.get_sql_dict()[
                    'delete_monthly_word_cloud_last_month'] if table.lower() == CONSTANTS.MONTHLY_WORD_CLOUD else \
                    MyUtils.get_sql_dict()[
                        'delete_monthly_words_verbatims_last_month'] if table.lower() == CONSTANTS.MONTHLY_WORDS_VERBATIMS else None

            conn = pg.connect(database=db_details['name'], user=db_details['user'],
                              password=db_details['password'], host=db_details['host'], port=db_details['port'])

            cursor = conn.cursor()

            logger.debug(' Read query = " %s "', sql)

            if sql is None:
                raise Exception("sql can not be 'None' type."
                                " No or wrong table name(s) passed to delete_previous_data() in MyUtils.py")

            cursor.execute(sql)

            conn.commit()

    except Exception:
        logger.error("Exception in delete_previous_data : %s ", traceback.format_exc())
        raise Exception
    finally:
        if cursor:cursor.close()
        if conn:conn.close()