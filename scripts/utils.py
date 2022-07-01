import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

class Utils:
    
    def __init__(self):
        pass

    def check_outlier(self, df):
        """
        calculates number of outliers found in each column of specified dataframe
        using interquiratile method

        Args:
            df: a dataframe with only numerical values
        
        Returns:
            a new dataframe with a count of minor and major outliers
        
        """
        tmp_info = df.describe()

        Q1 = np.array(tmp_info.iloc[4,:].values.flatten().tolist())
        Q3 = np.array(tmp_info.iloc[6,:].values.flatten().tolist())

        # calculate the Inerquartile range.
        IQR = Q3-Q1
        L_factor = IQR*1.5
        H_factor = IQR*3

        # Minor Outliers will lie outside the Inner fence
        Inner_Low = Q1-L_factor
        Inner_High = Q3 + L_factor
        inner_fence = [Inner_Low, Inner_High]

        # Major Outliers will lie outside the Outer fence
        Outer_Low = Q1-H_factor
        Outer_High = Q3+H_factor
        outer_fence = [Outer_Low, Outer_High]
        
        outliers = []
        for col_index in range(df.shape[1]):
            
            inner_count = 0
            outer_count = 0
            tmp_list = df.iloc[:,col_index].tolist()
            for value in tmp_list:
                if((value < inner_fence[0][col_index]) or (value > inner_fence[1][col_index])):
                    inner_count = inner_count + 1
                elif((value < outer_fence[0][col_index]) or (value > outer_fence[1][col_index])):
                    outer_count = outer_count + 1

            outliers.append({df.columns[col_index]:[inner_count, outer_count]})
        
        major_outlier = []
        minor_outlier = []
        columns = []
        outlier_dict = {}
        for item in outliers:
            columns.append(list(item.keys())[0])
            minor_outlier.append(list(item.values())[0][0])
            major_outlier.append(list(item.values())[0][1])

        outlier_dict["columns"] = columns
        outlier_dict["minor_outlier"] = minor_outlier
        outlier_dict["major_outlier"] = major_outlier
        outlier_df = pd.DataFrame(outlier_dict)

        return outlier_df


    def describe(self, df):
        """
        generates basic statistical information like mean, median, quartiles and others

        Args: 
            df: a dataframe that holds only numerical variables

        Returns:
            description: a dataframe that holds statistical information about the variables
        """
        description = df.describe().T.style.bar(subset=['mean'], color='#205ff2')\
                            .background_gradient(subset=['std'], cmap='Reds')\
                            .background_gradient(subset=['50%'], cmap='coolwarm')
        return description

    def normalize(self, df):
        """
        normalizes a dataframe by making the mean of each variable 0 and their SD 1

        Args:
            df: a dataframe that holds only numerical variables

        Returns:
            normal: a normalized dataframe.
        """
        normald = Normalizer()
        normal = pd.DataFrame(normald.fit_transform(df))
        return normal

    def scale(self, df):
        """
        scale variables using min-max scaler to bring all values between 0 and 1
        for each of the variables.

        Args:
            df: a dataframe that holds only numerical variables

        Returns:
            scaled: a dataframe with scaled variables.
        """
        scaler = MinMaxScaler()
        scaled = pd.DataFrame(scaler.fit_transform(df))
        return scaled

    def scale_and_normalize(self, df):
        """
        Runs the scaler and normalizer together and returns scaled and normalized 
        dataframe

        Args: 
            df: a dataframe with only numerical variables

        Returns: 
            normScaled: a dataframe with scaled and normalized variables 
        """
        columns = df.columns.to_list()
        normScaled = normalize(scale(df))
        normScaled.columns = columns
        return normScaled