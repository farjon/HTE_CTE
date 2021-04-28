'''
@@@ Letter Recognition Dataset @@@

Database of character image features;
try to identify the letter The first tabular dataset we examine is 'Letter Recognition' dataset and
the objective is to identify each of a large number of black-and-white rectangular
pixel displays as one of the 26 capital letters in the English alphabet.
The character images were based on 20 different fonts and each letter within these 20 fonts was
randomly distorted to produce a file of 20,000 unique stimuli.
Each stimulus was converted into 16 primitive numerical attributes (statistical moments and edge counts)
which were then scaled to fit into a range of integer values from 0 through 15.
We typically train on the first 16000 items and then use the resulting model to predict
the letter category for the remaining 4000.

@@@ Attribute Information @@@

lettr capital letter (26 values from A to Z)
x-box horizontal position of box (integer)
y-box vertical position of box (integer)
width width of box (integer)
high height of box (integer)
onpix total # on pixels (integer)
x-bar mean x of on pixels in box (integer)
y-bar mean y of on pixels in box (integer)
x2bar mean x variance (integer)
y2bar mean y variance (integer)
xybar mean x y correlation (integer)
x2ybr mean of x x y (integer)
xy2br mean of x y y (integer)
x-ege mean edge count left to right (integer)
xegvy correlation of x-ege with y (integer)
y-ege mean edge count bottom to top (integer)
yegvx correlation of y-ege with x (integer)
'''

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
General preparations
'''
desired_width = 1000
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 30)
data_path = "D:/Datasets/LETTER/LETTER.csv"

'''
General data analysis
'''
print("Letter data first few example")
data_letter = pd.read_csv(data_path)
print(data_letter.head())

print(data_letter.info())

print("Showing Statistical Summary")
statisticalSummary = data_letter.describe()
print(statisticalSummary)

print("Data Dimensions")
data_letter.shape

print("Mean for each letter")
data_mean_each_letter = data_letter.groupby("letter").mean()
print(data_mean_each_letter.head())

'''
Data Visualisation
'''
plt.figure(figsize=(13, 5))
sns.heatmap(data_mean_each_letter).set_title('Features effecting each letter')
plt.show()

order = list(np.sort(data_letter['letter'].unique()))
plt.figure(figsize=(13, 5))
sns.barplot(x='letter', y="xedge",
            data=data_letter,
            order=order).set_title('xedge feature for each letter')
plt.show()

# Correlation Matrix
# pip install Jinja2
plt.matshow(data_letter.corr())
plt.show()

# Scatterplot Matrix
# plt.figure(figsize=(13, 5))
# sns.set_theme(style="whitegrid")
# sns.pairplot(data_letter, hue="letter")
# plt.show()