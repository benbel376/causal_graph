import unittest
import numpy as np
import pandas as pd
import sys, os
 
# importing scripts
sys.path.insert(1, '..')
sys.path.append("..")
sys.path.append(".")

from scripts import data_viz
from scripts import data_cleaning
from scripts import data_transformation

DV = data_viz.Data_Viz("logs/test.log")
DC = data_cleaning.DataCleaner("logs/test.log")
DT = data_transformation.DataTransformer("logs/test.log")

class TestTweetDfExtractor(unittest.TestCase):
    """
		A class for unit-testing function in the fix_clean_tweets_dataframe.py file
		Args:
        -----
			unittest.TestCase this allows the new class to inherit
			from the unittest module
	"""

    def setUp(self) -> pd.DataFrame:
        test_dict = {"col1":[2, 4, 3, np.nan, np.nan, np.nan],
                    "col2":["02-04-2015", "03-01-2013", "03-11-2014", "10-02-2009", "03-11-2014", "10-09-2009"],
                    "col3":[1, 3, 10, np.nan, np.nan, np.nan]}
        self.test_df = pd.DataFrame(test_dict)
        # tweet_df = self.df.get_tweet_df()         


    def test_percent_missing(self):
        outcome = DV.percent_missing(self.test_df)
        test =  print("The dataset contains", round(
            (1/3), 2), "%", "missing values.")
        self.assertEqual(outcome, test)

if __name__ == '__main__':
	unittest.main()