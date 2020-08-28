import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

#data.info(); data.head()

def na_columns(data, threshold = 40, verbose = True):
    na_per = data.isna().sum()*100 / data.shape[0]
    if(verbose):
        print(na_per)
    return list(data.columns[na_per > threshold])

#drop columns with too many missing values

def plot_na_per(data, columns):
    plt.figure(num = None, figsize = (20, 5))
    sns.barplot(x = columns, y = data[columns].isna().sum()*100 / data.shape[0])
    plt.xticks(rotation = 45, horizontalalignment = 'right', fontweight = 'light')
    plt.title("% of NAs in each column")
    plt.show()

def num_unique_vals_cols(data, variables):
    num_unique_levels = []
    for variable in variables:
        num_levels = data[variable].nunique()
        num_unique_levels.append(variable + ": " + str(data[variable].dtype) + " " + str(num_levels))
    return num_unique_levels

def drop_single_value_cols(data):
    drop_columns = data.columns[data.nunique() == 1]
    data.drop(drop_columns, axis = 1, inplace = True)
    return data

#use this to identify right types, remove columns with single unique values, identify cat columns with huge levels
#if a column has a lot of na values and a single value, do not remove that column
#change features from one type to another if only minor manipulation is needed

#use frequency_counts to assess data types

def convert_type(data, variables, target_type):
    #targettype can be int64, float64, datetime64, category, int32 etc
    for variable in variables:
        data[variable] = data[variable].astype(target_type)
    return data

#change date format
#data["day"] = pd.to_datetime(data.session_start, format = "%Y-%m-%d %H:%M:%S")
def date_to_features(data, variables):
    #day, month, year, week
    return "dada"

def find_cat_num_cols(data):
    num_search_term = "^float|^int"
    cat_search_term = "^object|^category"
    num_cols = list(data.columns[data.dtypes.apply(lambda x: re.search(num_search_term, str(x)) != None)])
    cat_cols = list(data.columns[data.dtypes.apply(lambda x: re.search(cat_search_term, str(x)) != None)])
    return num_cols, cat_cols

############ Univariate #################

def frequency_counts(data, variables):
    for variable in variables:
        print(data[variable].value_counts(normalize = True))
        
#use this to identify binnable categorical variables, lopsided distribution of levels - measure of lopsided ness???
#remove cat columns that have too much data corresponding to a single level(98% for example)
#remove cat columns if there are too many levels except the id (less than x %obs per level, high dimensional sparse data)

def bin_cat_cols(data, variables, thresholds):
    for variable, threshold in zip(variables, thresholds):
        frequency_counts = data[variable].value_counts(normalize = True)
        group_levels = frequency_counts.index[frequency_counts < threshold]
        filter = data[variable].isin(group_levels)
        data["binned_" + variable] = data[variable]
        data["binned_" + variable][filter] = "grouped_" + variable
        
    return data

def response_rate_bins(data, variables):
    """return data with levels of a cat col binned according to response rate of each level"""
    return "dada"

#create list of plottable cat columns with manageable levels, use that for the next function
    
def plot_cat_columns(data, variables, num_rows, num_cols, figsize = (20, 10)):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    plt.suptitle("Barplot of categorical features")

    i = 1
    for variable in variables:
      plt.subplot(num_rows, num_cols, i)
      temp_series = data[variable].value_counts()  
      sns.barplot(temp_series.index, temp_series)
      plt.title(variable)
      plt.xticks(rotation=45, horizontalalignment='right', fontweight='light')
      i = i + 1

    plt.show()

#data.describe()
#check data values, negative, outliers, mean, median
#if std is close to zero remove them

def is_outlier_median(points, thresh=4):
   """
   References:
   ----------
   Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
   Handle Outliers", The ASQC Basic References in Quality Control:
   Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
   """
   if len(points.shape) == 1:
      points = points[:,None]
   median = np.median(points, axis=0)
   diff = np.sum((points - median)**2, axis=-1)
   diff = np.sqrt(diff)
   med_abs_deviation = np.median(diff)
   
   modified_z_score = 0.6745 * diff / med_abs_deviation
   return modified_z_score > thresh

#univariate outlier analysis - remove outliers; fill with mean, median; make na;treat them apart - make segments if they are too many

def plot_hist_num(data, variables, num_rows, num_cols, figsize, outlier_param = 3, remove_outlier = True):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    i = 1
    for variable in variables:
        plt.subplot(num_rows, num_cols, i)
        plot_data = data[variable].dropna()
        if(remove_outlier):
            plt.suptitle("Histgoram of numerical features (stripped of outliers)")
            plt.hist(plot_data[~is_outlier_median(plot_data, outlier_param)])
        else:
            plt.suptitle("Histgoram of numerical features")
            plt.hist(plot_data)
        plt.title(variable)
        i = i + 1
    plt.show()
    
    
################# Bivariate ##############

#kepp relevant columns for plotting bivariate association, exclude id and dates
#dython.nominal.associations(data[all_cols, clustering = True, figsize = (20, 20)])
#keep one of the highly correlated variables

################# Independent dependent relationship(1-1) ##############

def plot_num_numout(data, variables, outcome_col, num_rows, num_cols,  figsize, alpha, outlier_param = 3, remove_outlier = True):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    plt.suptitle("Relationship of numerical features with numerical outcome")

    i = 1
    for variable in variables:
        plt.subplot(num_rows, num_cols, i)
        plotting_data = data[[variable, outcome_col]].dropna()
        if(remove_outlier):
            filter1 = is_outlier_median(plotting_data[variable], outlier_param) # change this to multivariate outlier detection
            filter2 = is_outlier_median(plotting_data[outcome_col], outlier_param)
            sns.scatterplot(plotting_data[variable][~(filter1 | filter2)], plotting_data[outcome_col][~(filter1 | filter2)], alpha = alpha)
        else:
            sns.scatterplot(plotting_data[variable], plotting_data[outcome_col], alpha = alpha)
        plt.title(variable)
        i = i + 1

    plt.show()
    
def plot_cat_numout(data, variables, outcome_col, num_rows, num_cols, figsize, outlier_param = 3, remove_outlier = True):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    plt.suptitle("Relationship of categorical features with numerical outcome")

    i = 1
    for variable in variables:
        plt.subplot(num_rows, num_cols, i)
        plot_data = data[[variable, outcome_col]].dropna()
        if(remove_outlier):
            filter = is_outlier_median(plot_data[outcome_col], outlier_param)
            sns.violinplot(plot_data[variable][~filter], plot_data[outcome_col][~filter])
        else:
            sns.violinplot(plot_data[variable], plot_data[outcome_col])
        plt.title(variable)
        i = i + 1

    plt.show()
    
def plot_num_catout(data, variables, outcome_col, num_rows, num_cols, figsize, outlier_param = 3, remove_outlier = True):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    plt.suptitle("Relationship of numerical features with categorical outcome")

    i = 1
    for variable in variables:
        plt.subplot(num_rows, num_cols, i)
        plot_data = data[[variable, outcome_col]].dropna()
        if(remove_outlier):
            filter = is_outlier_median(plot_data[variable], outlier_param)
            sns.violinplot(plot_data[outcome_col][~filter], plot_data[variable][~filter])
        else:
            sns.violinplot(plot_data[outcome_col], plot_data[variable])
        plt.title(variable)
        i = i + 1

    plt.show()
    
def plot_cat_catout(data, variables, outcome_col, num_rows, num_cols, figsize):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    plt.suptitle("Relationship of categorical features with categorical outcome")

    i = 1
    for variable in variables:
        plt.subplot(num_rows, num_cols, i)
        sns.countplot(x = data[variable], hue = data[outcome_col])
        i = i + 1

    plt.show()

################# Multivariate relationships ##############
#sns.scatterplot(data[input_col], data[outcome_col], hue = hue, size = size, style = style, alpha = alpha)
    

################## Feature Engineering #####################

#### change the distribution ####
#log, square root, cube root, non linear to linear, to normal, to uniform
"""
pt = sklearn.preprocessing.PowerTransformer(method='box-cox', standardize=True)
pt = sklearn.preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
pt = sklearn.preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
data2 = pt.fit_transform(data1)
"""

#### Scale numerical features ####
"""
scaled_data = sklearn.preprocessing.StandardScaler().fit_transform(data)
scaled_data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)
scaled_data = sklearn.preprocessing.RobustScaler().fit_transform(data) - robust to outliers
scaled_data = sklearn.preprocessing.MaxAbsScaler().fit_transform(data) - maintain sparsity
"""

#### Polynomial features ####
"""
poly = sklearn.preprocessing.PolynomialFeatures(2, interaction_only=False)
poly.fit_transform(X)
"""

#### Numerical encoding of categorical variables ####
"""
encoder = category_encoders.hashing.HashingEncoder(n_components = 8)
train2 = encoder.fit_transform(train, y)

encoder = category_encoders.one_hot.OneHotEncoder(cols = encoding_cat_cols)
train2 = encoder.fit_transform(train, y)
"""

#Binning of categorical feature

#### Dimensionality reduction and clustering #####
# visualise clusters to see if there is need to fit multiple models

"""
tsne = sklearn.manifold.TSNE(n_components= 2)
data2 = tsne.fit_transform(data1)
sns.scatterplot(data2[:, 0], data2[:, 1]) - plot data2 to see if there are clusters of outliers

pca = sklearn.decomposition.PCA(n_components = 2)
data2 = pca.fit_transform(data1)

"""

#### Outlier treatment ######

#multivariate outlier analysis - remove outliers; fill with mean, median; make na;treat them apart - make segments if they are too many

"""
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

outliers_fraction = 0.15
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))]

for name, algorithm in anomaly_algorithms:
    if name == "Local Outlier Factor":
        y_pred = algorithm.fit_predict(X)
    else:
        y_pred = algorithm.fit(X).predict(X)
        
"""

def is_outlier_num_cols(data, variables, outlier_params):
    for variable, outlier_param in zip(variables, outlier_params):
        data[variable + "_is_outlier"] = is_outlier_median(data[variable], outlier_param)
    return data

#### Treating Missing Values #########
#choice of imputing data after or before outlier treatment
"""
data.dropna() - if missing is completely random then one can delete rows with missing values if we have lots of observation
data[cat_cols] = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value = "unknown").fit_transform(data[cat_cols])
    #this is not good as it would change the relationship of th said variable with other variables
data[cat_cols] = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(data[cat_cols])
data[num_cols] = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(data[num_cols]), case of predicting with only constant as independent variable,
    #if there are too many missing(more than 20%) values, this may distort the pattern
"""

#prediction model for imputation, best to do this as missing values may depend on other variables (not missing completely at random)
    #data more well behaved then usual, in absence of relationship with other variables, imputation is not good
#multivariate imputer
"""
from sklearn.experimental import enable_iterative_imputer
imp = sklearn.impute.IterativeImputer(max_iter=10, random_state=0, initial_strategy = 'median')
data_total = imp.fit_transform(data_total)
"""

#### binning numerical variables ####
#use this when different bins of numerical variables show varying linear pattern with target 
"""
train1 = sklearn.preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='onehotdense', strategy = "kmeans").fit_transform(train)
train1 = sklearn.preprocessing.Binarizer().fit_transform(train)
"""

#difference in date, time, addresses; ratio proportion of variables
#seasonality
#convert range variables to upper and lower bound variables
#covariate binning

################### Miss ##############################
"""
pd.concat((data_total, data_temp)) - row concat
data.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)
data[variable].astype("timedelta64[s]")
pd.pivot_table(data1, index = ["player_id"], columns = ['software_id'], values = "payout_converted", aggfunc = np.sum) - long to wide format
data.groupby(variable1).agg({variable2: ['sum', 'min']})
data.groupby("variable1").agg({'variable2': 'sum'}).reset_index().sort_values(by = 'variable2', ascending = False).rename(columns={variable2: variable3})
df.min(axis = 0), columnwise minimum
new_data = data.groupby([variable1, variable2]).apply(func, func_param1, func_param2).reset_index()
data[new_col] = data.groupby([variable1, variable2]).apply(func, func_param1, func_param2).values - func returns a series
data.iloc[:n,]
data[variable][:n,]
data.iloc[[1, 2, 3], [0, 1]]
data.loc[boolean_filter, boolean_filter]
[x in y_true for x in y_pred] - for numpy and list
data[data[variable1].isin(list_of_values)]
pd.DataFrame({"n_items": [n_items], "n_correct": [n_correct], "per_correct": [per_correct]})

plt.figure(figsize = (20, 5))
plt.box(False)
plt.grid(linewidth = 1, alpha = 0.3)
plt.tick_params(axis='both', which='both', bottom=False, top=False, right = False, left = False)
plt.legend(frameon = False)

"""

"""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize = (20, 5))
plt.box(False)
plt.grid(linewidth = 1, alpha = 0.3)
plt.tick_params(axis='both', which='both', bottom=False, top=False, right = False, left = False)

sns.lineplot(x = quantiles, y = model_perf, label = "CF Model")
sns.lineplot(x = quantiles, y = pop_perf, label = "Popularity Model")
plt.xlabel("Percentile above which popular games are excluded")
plt.ylabel("Rank")
plt.title("Both popular and CF models are to be used; For CF model use data where popular games above 90 percentile are excluded.")

plt.legend(frameon = False)
plt.show()
"""