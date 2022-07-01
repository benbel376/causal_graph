import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer, MinMaxScaler
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
        normScaled = self.normalize(self.scale(df))
        normScaled.columns = columns

        return normScaled


    def remove_correlated(self, df, th):
        """
        removes highly correlated variables from a dataframe.

        Args:
            df: a features dataframe that holds the variables
            th: a threshold correlation value to decide which variables to remove

        Return:
            features_df: a new features dataframe with low correlation values. 
        """
        corrmat = df.corr()
        correlated_features = set()
        for i in range(len(corrmat.columns)):
            for j in range(i):
                if abs(corrmat.iloc[i, j]) >= th:
                    colname = corrmat.columns[i]
                    correlated_features.add(colname)

        print(f"number of correlated variables: {len(correlated_features)}")
        print("..................................................")
        print("correlated features: ", correlated_features)

        features_df = df.drop(labels=correlated_features, axis=1)

        return features_df


    def select_features_RFE(self, features_r, target_r, num):
        """
        filters features using the Recurssive Feature Elimination method
        that applies randomforest regressor as estimator

        Args:
            features_r: a dataframe of features with unscaled and unnormalized values
            target_r: a series that contains target value in string form.
            num: number of features to return

        Returns:
            new_features: a dataframe of selected features.
        """
        features = StandardScaler().fit_transform(features_r)
        target = LabelEncoder().fit_transform(target_r)
        # Init the transformer
        rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=num)

        # Fit to the training data
        _ = rfe.fit(features, target)

        # extract features
        new_features = features_r.loc[:, rfe.support_]

        return new_features


    # random forest checker
    def forest_test(self, features_r, target_r):
        """
        checkes the target prediction accuracy of a given set of features
        and prints the accuracy.

        Args:
            features_r: features dataframe that is not scaled or normalized
            target_r: target dataframe that is not encoded

        Returns: None
        
        """
        features = StandardScaler().fit_transform(features_r)
        target = LabelEncoder().fit_transform(target_r)

        X_Train, X_Test, Y_Train, Y_Test = train_test_split(features, target, 
                                                            test_size = 0.30, 
                                                            random_state = 11)
        forest = RandomForestClassifier(n_estimators=700)
        _ = forest.fit(X_Train, Y_Train)
        print(f"accuracy score: {forest.score(X_Test, Y_Test)}")


    def apply_treshold(sm, th):
        """
        removes edges from a structure model based on provided treshold value

        Args:
            sm: causalnex structure model with nodes and edges
            th: a weight treshold to use as a reference to remove edges

        Return:
            sm_copy: a new causalnex structure model with some  week edges removed.
        
        """
        sm_copy = copy.deepcopy(sm)
        sm_copy.remove_edges_below_threshold(th)

        return sm_copy
    
    def plot_graph(sm, th, title, name=None, save=False):
        """
        plots a structure model or causal graph by not including edges below the th.

        Args:
            sm: a causalnex structure model
            th: a treshold to use as a reference to eleminate some week edges.
            title: title for the image

        Returns: Image object that holds the causal graph

        """
        path = f"../data/images/{name}"
        tmp = apply_treshold(sm, th)
        viz = plot_structure(
            tmp,
            title=title,
            graph_attributes={"scale": "2.5", 'size': 2},
            all_node_attributes=NODE_STYLE.WEAK,
            all_edge_attributes=EDGE_STYLE.WEAK)
        img = Image(viz.draw(format='png'))
        
        if(save):
            path = f"../data/images/{name}"
            img.save(path)

        return img

    def jacc_index(sm1, sm2, th1, th2, formatted=True):
        """
        calculates jaccard similarity index between two causal graphs.

        Args:
            sm1: causal graph 1
            sm2: causal graph 2
            th1: threshold for first graph for elementation
            th2: threshodl for second graph
            formatted: weather to reurn a formated text or just index value

        Returns:
            sim: a similarity index
            text: a formated information.
        """
        sm1_copy = copy.deepcopy(sm1)
        sm2_copy = copy.deepcopy(sm2)
        sm1_copy.remove_edges_below_threshold(th1)
        sm2_copy.remove_edges_below_threshold(th2)
        a = sm1_copy.edges
        b = sm2_copy.edges
        n = set(a).intersection(b)
        sim = round(len(n) / (len(a) + len(b) - len(n)), 2)
        if(formatted):
            return f"The similarity index: {sim}"
        else:
            return sim 


    def corr(self, x, y, **kwargs):
        """
        calculates a correlation between two variables

        Args:
            x: a list of values
            y: a list of values

        Returns: nothing
        """
        # Calculate the value
        coef = np.corrcoef(x, y)[0][1]
        # Make the label
        label = r'$\rho$ = ' + str(round(coef, 2))
        
        # Add the label to the plot
        ax = plt.gca()
        ax.annotate(label, xy = (0.2, 0.95), size = 11, xycoords = ax.transAxes)

        
    def plot_pair(self, df, title, range, size, save=False, name=None):
        """
        generates a pair plot that shows distribution of one variable and 
        its relationship with other variables using scatter plot.

        Args:
            range: the range of variables to include in the chart
            size: the size of the chart

        Returns: None.
        """
        target = df["diagnosis"]
        data = df.iloc[:,1:]
        data = pd.concat([target,data.iloc[:,range[0]:range[1]]],axis=1)
        plt.figure(figsize=(size[0],size[1]))
        plt.title(title)
        grid=sns.pairplot(data=data,kind ="scatter",hue="diagnosis",palette="Set1")
        grid = grid.map_upper(self.corr)
        

        if(save):
            path = f"../data/images/{name}"
            plt.savefig(path)


    
    def show_corr(self, df, title, size=[17,10], range=None, save=False, name=None):
        """
        plots a correlation matrix heatmap

        Args:
            df: dataframe that holds the data
            size: size of the chart to be plotted
            range: the range of columns or variables to include in the chart

        Returns: None
        """
        # correlation matrix
        if range is None:
            corr_matrix = df.corr()
        else:
            if(range[1] == -10):
                corr_matrix = df.iloc[:,range[0]:].corr()
            else:
                corr_matrix = df.iloc[:,range[0]:range[1]].corr()
        matrix = np.triu(corr_matrix)
        fig, ax = plt.subplots(figsize=(size[0], size[1]))
        plt.title(title)
        ax = sns.heatmap(corr_matrix, annot=True, mask=matrix)
        

        if(save):
            path = f"../data/images/{name}"
            plt.savefig(path)



    def plot_violin(self, df, size, title, save=False, name=None):
        """
        plots a violin graph

        Args:
            df: a dataframe that holds both the feature and target variables
            size: a list that holds the size of the chart to be plotted
            save: whether to savethe data or not.
            name: name of the chart to save.

        Returns: None
        """
        df.iloc[:,1:] = self.scale_and_normalize(df.iloc[:,1:]) 
        data = pd.concat([df.iloc[:,:]],axis=1)
        data = pd.melt(data,id_vars="diagnosis",
                            var_name="features",
                            value_name='value')
        plt.figure(figsize=(size[0],size[1]))
        plt.title(title)
        sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart",palette ="Set2")
        plt.xticks(rotation=90)

        if(save):
            path = f"../data/images/{name}"
            plt.savefig(path)