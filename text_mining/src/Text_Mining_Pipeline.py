import os
import sys
import logging
import traceback
from datetime import datetime
# from langdetect import detect
from text_mining.src.utility import Dao
from text_mining.src.utility import MyUtils
from text_mining.definitions import ROOT_DIR
from text_mining.src.utility import CONSTANTS
from text_mining.src import Classification_Helper
from logging.handlers import RotatingFileHandler
from dateutil.relativedelta import relativedelta


def text_mining_pipeline(args, log_dir):
    LOGGING_LEVEL = logging.DEBUG

    LOG_FILE_NAME = os.path.join(log_dir, 'TEXT_MINING_LOG.out')
    logger = logging.getLogger(__name__)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s ')
    handler = RotatingFileHandler(LOG_FILE_NAME, maxBytes=10 * 1024, backupCount=5)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    dao_logger, utils_logger = Dao.logger, MyUtils.logger

    logger.setLevel(LOGGING_LEVEL)
    dao_logger.setLevel(LOGGING_LEVEL)
    utils_logger.setLevel(LOGGING_LEVEL)

    dao_logger.addHandler(handler)
    utils_logger.addHandler(handler)

    start, max_date_month = None, None

    print('\n \n Check LOG file for status \n \n ')

    if len(args) == 1:
        logger.error("\n No command line argument passed ! \n"
                     " \n YOU are supposed to pass one script parameter \n"
                     " \n Possible values -> \n 1. 'train'  2. 'all'  3. 'update' \n \n ")
        sys.exit(1)

    elif args[1].lower() not in [CONSTANTS.TRAIN, CONSTANTS.ALL, CONSTANTS.UPDATE]:
        logger.error("Wrong parameter passed \n  \n Possible values -> \n 1. 'train'  2. 'all'  3. 'update' \n \n ")
        sys.exit(1)

    elif args[1].lower() == CONSTANTS.TRAIN:
        ''' Currently reading data to train from file. But Already have function to read from database.'''

        logger.info('\n \n Running Pipeline for')
        logger.info(CONSTANTS.TRAIN.upper())

        X, y = MyUtils.get_train_data_from_file()
        Classification_Helper.train_model(MyUtils.process_data(data=X, type=None), y)
        logger.info('Training for model done')
        # Training part of Pipeline ends here

    else:
        try:
            if args[1].lower() == CONSTANTS.ALL:
                '''ALL option deletes all data from monthly table then runs pipeline from month of minimum date
                to current month - 1 '''

                logger.info('\n \n Running Pipeline for')
                logger.info(CONSTANTS.ALL.upper())

                Dao.delete_previous_data(tables=[CONSTANTS.MONTHLY_SENTIMENT, CONSTANTS.MONTHLY_WORD_CLOUD,
                                                 CONSTANTS.MONTHLY_WORDS_VERBATIMS], period=CONSTANTS.ALL)

                min_date = Dao.get_min_data_date()
                start = min_date.replace(day=1)

            elif args[1].lower() == CONSTANTS.UPDATE:
                ''' UPDATE runs pipeline from next month of last entry date till current month - 1 '''

                logger.info('\n \n Running Pipeline for')
                logger.info(CONSTANTS.UPDATE.upper())

                start = Dao.get_last_updated_date(type=CONSTANTS.PREVIOUS)
                # start = start.replace(day=1) + relativedelta(months=1)
                start = start.replace(day=1)

            # if datetime.now().month == 1:
            #     max_date_month = 12
            #     max_date_year = datetime.now().year - 1
            # else:
            #     max_date_month = datetime.now().month - 1
            #     max_date_year = datetime.now().year
            #
            # max_date = datetime(year=max_date_year, month=max_date_month, day=1).date()
            max_date = datetime.now().date().replace(day=MyUtils.get_last_date_of_month(datetime.now().date()).day)

            if start <= max_date:
                # start = start - relativedelta(months=1)
                logger.info(
                    'Deleting, if present, and running for %s' % start.strftime(
                        "%m-%Y"))

                Dao.delete_previous_data(tables=[CONSTANTS.MONTHLY_SENTIMENT, CONSTANTS.MONTHLY_WORD_CLOUD,
                                                 CONSTANTS.MONTHLY_WORDS_VERBATIMS], period=CONSTANTS.PREVIOUS)

                if datetime.now().day <= 10 and start.month == max_date.month:
                    logger.info('Date in first 10 days... So deleting for one more previous month')

                    Dao.delete_previous_data(tables=[CONSTANTS.MONTHLY_SENTIMENT, CONSTANTS.MONTHLY_WORD_CLOUD,
                                                     CONSTANTS.MONTHLY_WORDS_VERBATIMS], period=CONSTANTS.PREVIOUS)
                    start = start - relativedelta(months=1)

            while start <= max_date:
                end = MyUtils.get_last_date_of_month(start)
                new_data = Dao.get_new_data_date_range(start, end)

                if len(new_data) > 0:
                    texts, issue_1, issue_3, agent_ids, dates, sids = zip(*new_data)

                    to_be_inserted = []
                    last_date = MyUtils.get_last_date_of_month(date=start)
                    month_year = MyUtils.get_formatted_month_year_name(date=last_date)
                    cleaned_data = MyUtils.process_data(data=texts, type=None)
                    predicted = Classification_Helper.classify_new_data(x_new=cleaned_data)

                    for verbatim, sentiment, lvl1, lvl3, agent_id, survey_date, sid in zip(texts, predicted, issue_1,
                                                                                           issue_3,
                                                                                           agent_ids, dates, sids):
                        to_be_inserted.append(
                            (last_date, verbatim, sentiment, lvl1, lvl3, agent_id, month_year, survey_date, sid))

                    # send list of tuples to insert
                    Dao.insert_into_monthly_sentiment(data=to_be_inserted)

                    for sentiment_type in [CONSTANTS.POSITIVE, CONSTANTS.NEGATIVE]:
                        one_sentiment_data = MyUtils.separate_single_sentiment_data(data=zip(texts, predicted),
                                                                                    sentiment=sentiment_type)

                        cleaned_data_no_adj_adv = MyUtils.process_data(data=one_sentiment_data,
                                                                       type=CONSTANTS.NO_ADJ_ADV)

                        words_list, word_cloud_dict = MyUtils.get_word_cloud_data(
                            data=cleaned_data_no_adj_adv,
                            sentiment=sentiment_type)

                        verbatims_for_word_data = MyUtils.get_verbatims_for_word(data=one_sentiment_data,
                                                                                 words=words_list,
                                                                                 sentiment=sentiment_type,
                                                                                 last_date=last_date,
                                                                                 month_year_name=month_year,
                                                                                 agent_id=CONSTANTS.ANALYST)

                        word_cloud_data = MyUtils.prepare_word_cloud_data(last_date=last_date,
                                                                          verbatims_for_words=verbatims_for_word_data,
                                                                          word_cloud_dict=word_cloud_dict,
                                                                          sentiment=sentiment_type,
                                                                          nt_id=CONSTANTS.ANALYST,
                                                                          month_year=month_year)

                        Dao.insert_word_cloud_data(data=word_cloud_data)

                        Dao.insert_monthly_words_verbatims(words_verbatims_data=verbatims_for_word_data)

                start = start + relativedelta(months=1)

        except Exception:
            logger.error('\n Exception occurred for %s \n ' % traceback.format_exc())
            logger.error('Exception occurred for %s \n \n ' % sys.argv[1])
            sys.exit(1)
        else:
            logger.info('" %s " option processes done \n\n' % sys.argv[1].upper())
