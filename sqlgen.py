import tgym
import numpy as np
import pandas as pd
# import sklearn.preprocessing as skp
import json
import sqlalchemy
import datetime as dt
import random


def randDates(start=2014, end=2017, days=3):
    """
    Returns a pair of random stretch of business days between start and end year
    start: start date
    end: end date
    days: stretch of days we're looking for
    """
    if start == 2014:
        start = dt.date(start, 3, 3)  # 2014 data starts from 3rd march
    else:
        start = dt.date(start, 1, 1)
    end = dt.date(end, 12, 31)
    endReduced = end - pd.tseries.offsets.BDay(days)  # we need to leave atleast enough gap as the num of days
    dayRange = len(pd.date_range(start, endReduced, freq=pd.tseries.offsets.BDay()))
    rndStart = np.floor(random.random() * dayRange)
    rndStartDate = start + pd.tseries.offsets.BDay(rndStart)
    rndEndDate = rndStartDate + pd.tseries.offsets.BDay(days)

    return rndStartDate, rndEndDate


class SQLStreamer(tgym.DataGenerator):
    """Data generator from csv file.
        configFile: Location of json file containing database information
        numDays: Number of days at a stretch to be used to generate data
        YearBegin: Date from when to begin gathering data
        YearEnd: Date from when to stop gathering data (inclusive)
        scrip: ticker of scrip to gather data about

        yields numpy array containing price from this iteratons
        """
    @staticmethod
    def _generator(configFile='config.json', numDays=3, YearBegin=2014, YearEnd=2017, scrip='RELIANCE'):
        """
        Connects to a postgres database and then
        iterates over a pandas dataframe to return the close price
        Might miss the point of using generators
        """
        start, end = randDates(start=2014, end=2017, days=numDays)
        address = json.load(open('config.json'))['DB']['ADDRESS']
        engine = sqlalchemy.create_engine(address)
        query = "SELECT * FROM histdata \
                 WHERE ticker='%s' \
                 AND datetime BETWEEN '%s' \
                 AND '%s' \
                 ORDER BY datetime ASC" % (scrip, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        dat = pd.read_sql(query, engine, parse_dates=['datetime'])
        dat.index = pd.DatetimeIndex(dat.datetime)
        dat = dat.tz_convert('Asia/Kolkata')
        closeD = dat['close'].values
        # with open(filename, "rb") as csvfile:
        #     reader = csv.reader(csvfile)
        #     if header:
        #         next(reader, None)
        #     for row in reader:
        #         assert len(row) % 2 == 0
        #         yield np.array(row, dtype=np.float)
        for price in closeD:
            yield np.array(price, dtype=np.float)

    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        print ("End of data reached, rewinding.")
        super(self.__class__, self).rewind()

    def rewind(self):
        """For this generator, we want to rewind only when the end of the data is reached.
        """
        pass
