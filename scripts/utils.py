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