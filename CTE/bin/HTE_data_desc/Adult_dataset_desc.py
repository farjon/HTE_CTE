'''
@ Adult Dataset @
Predict whether income exceeds $50K/yr based on census data.
Also known as "Census Income" dataset.
Extraction was done by Barry Becker from the 1994 Census database.
A set of reasonably clean records was extracted using the following conditions:
((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
Prediction task is to determine whether a person makes over 50K a year.
@@@@@@@@@@@@@@@@@@

### Attribute Information ####

Attribute Information
50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education:  Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters,
            1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent,
                Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners,
            Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv,
            Armed-Forces,unemployed.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India,
                Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
                Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary,
                Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong,
                Holand-Netherlands.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale

'''
Functions
'''


def check_nans(df_nan):
    for i in range(df_nan.shape[1]):
        n_miss = df_nan.iloc[:, i].isnull().sum()
        perc = n_miss / df_nan.shape[0] * 100
        col_name = df_nan.columns[i]
        print('> %d (%s), Missing: %12d values (%.1f%%)' % (i, col_name, n_miss, perc))


'''
Classes
'''


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.
        >   Columns of dtype object are imputed with the most frequent value
            in column.
        >   Columns of other types are imputed with mean of column.
        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


'''
General preparations
'''
desired_width = 1000
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)
data_path = "D:/Datasets/ADULT/ADULT.csv"

'''
General data analysis
'''
print("\n \n \nAdult data first few example")
# data first few example
data_adult = pd.read_csv(data_path)
print(data_adult.head())
print("\n \n \n")

print("Shape and dimensions of the data is:".format(data_adult.info()))

print(data_adult.shape)

statisticalSummary = data_adult.describe()
print(statisticalSummary)

'''
Re-arranging the data
'''
data_adult = data_adult.replace(regex=r'\s', value="")  # remove all spaces from table values
print((data_adult == 0).astype(int).sum())  # Counts how many zeros in each column
# Relative to the rest of the rows
capital_gain_zeros = (data_adult["capital.gain"] == 0).sum()
capital_loss_zeros = (data_adult["capital.loss"] == 0).sum()
total_rows = len(data_adult.index)
print("\n{:.3f}% of all rows of capital.gain are zeros".format((capital_gain_zeros / total_rows) * 100))
print("{:.3f}% of all rows of capital.loss are zeros\n".format((capital_loss_zeros / total_rows) * 100))
# Dropping capital.loss and capital.gain columns because there are too many zeros... nothing to learn here
del data_adult["capital.gain"]
del data_adult["capital.loss"]
print("Checking duplicated values:")
if data_adult.duplicated().any():
    print("FIY there are duplicated rows in the data\n")
else:
    print("FIY there are NO duplicated rows in the data\n")
# print("Dropping duplicated values and checking again")
# data_adult = data_adult.drop_duplicates()
# if data_adult.duplicated().any():
#     print("There are duplicated rows in the data\n")
# else:
#     print("There are NO duplicated rows in the data\n")
print("Check for missing values:")
isMissing = data_adult.isnull().sum().sum()
if isMissing:
    print(f'There are {isMissing} missing values in the data\n')
else:
    print(f"None missing values - {isMissing} missing values in the data\n")

df_nan = data_adult.replace(regex=r'["?"]', value=np.nan)  # replace ? with Nan
print("Columns that contains NANs\n-----------------")
check_nans(df_nan)
print(
    f"\nNans summary\n---------\n{df_nan['workclass'].isnull().sum()} workclass Nans\n{df_nan['occupation'].isnull().sum()} occupation Nans\n{df_nan['native.country'].isnull().sum()} native.country Nans.")
print("\n\t**We filled occupation as 'unemployed' for those who 'never-worked' in their workclass feature")
df_nan = df_nan.drop(df_nan[df_nan["workclass"] == r'Never-worked'].index)
print("\nColumns that contains NANs\n-----------------")
check_nans(df_nan)
print(
    f"\nNans summary\n---------\n{df_nan['workclass'].isnull().sum()} workclass Nans\n{df_nan['occupation'].isnull().sum()} occupation Nans\n{df_nan['native.country'].isnull().sum()} native.country Nans.")

# print(f"\n\t**We've deleted all {df_nan['workclass'].isnull().sum()} rows where occupation and workclass features are "
#       f"Nans\n\n")
# df_nan_no_occupation_workclass = df_nan.drop(df_nan[df_nan["workclass"].isnull()].index)  # delete Nans from occupation and workclass


# fit on the dataset
df_filled = DataFrameImputer().fit_transform(df_nan)
print("\n\nAfter fitting missing values\n-------------------")
check_nans(df_filled)
'''
Encoding Categorical Values
'''
#### One Hot many cases
print("\n\n\nAfter One Hot(for all cases) Encoding\n-----------------------------------")
print(f"shape before one hot is {df_filled.shape}")
one_hot_enc = OneHotEncoder(sparse=False)
df_filled_ohe_all_cases = (one_hot_enc.fit_transform(df_filled))
print(f"shape after one hot is is {df_filled_ohe_all_cases.shape}")

#### One Hot Binary
print("\n\n\nAfter One Hot Binary Encoding\n-----------------------------------")
df_filled_OH = df_filled
df_filled_OH = pd.get_dummies(df_filled,
                               columns=["workclass", "education", "marital.status", "occupation", "relationship",
                                        "race", "sex",
                                        "native.country", "income"],
                               prefix=["workclass", "education", "marital.status", "occupation", "relationship", "race",
                                       "sex", "native.country", "income"])
print(df_filled_OH.head(30))

###  Label Encoding
print("\n\n\nAfter Label Encoding\n-----------------------------------")
df_filled_labeling_encoding = df_filled
df_filled_labeling_encoding["workclass"] = df_filled["workclass"].astype('category')
df_filled_labeling_encoding["education"] = df_filled["education"].astype('category')
df_filled_labeling_encoding["marital.status"] = df_filled["marital.status"].astype('category')
df_filled_labeling_encoding["occupation"] = df_filled["occupation"].astype('category')
df_filled_labeling_encoding["relationship"] = df_filled["relationship"].astype('category')
df_filled_labeling_encoding["race"] = df_filled["race"].astype('category')
df_filled_labeling_encoding["sex"] = df_filled["sex"].astype('category')
df_filled_labeling_encoding["native.country"] = df_filled["native.country"].astype('category')
df_filled_labeling_encoding["income"] = df_filled["income"].astype('category')

df_filled_labeling_encoding["workclass.cat"] = df_filled["workclass"].cat.codes
df_filled_labeling_encoding["education.cat"] = df_filled["education"].cat.codes
df_filled_labeling_encoding["marital.status.cat"] = df_filled["marital.status"].cat.codes
df_filled_labeling_encoding["occupation.cat"] = df_filled["occupation"].cat.codes
df_filled_labeling_encoding["relationship.cat"] = df_filled["relationship"].cat.codes
df_filled_labeling_encoding["race.cat"] = df_filled["race"].cat.codes
df_filled_labeling_encoding["sex.cat"] = df_filled["sex"].cat.codes
df_filled_labeling_encoding["native.country.cat"] = df_filled["native.country"].cat.codes
df_filled_labeling_encoding["income.cat"] = df_filled["income"].cat.codes
print(df_filled.head(20))

'''
Data Visualisation
'''
# Correlation Matrix
# pip install Jinja2
'''
TODO: change data to
data = {'A': [45,37,42,35,39],
        'B': [38,31,26,28,33],
        'C': [10,15,17,21,12]
        }
'''
# df = pd.DataFrame(data_adult)
# corrMatrix = df.corr()
# corrMatrix.style.background_gradient(cmap='coolwarm').set_precision(4)
# plt.matshow(corrMatrix)
# plt.show()
