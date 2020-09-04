#import dependencies
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

#load in data
nba = pd.read_csv('nba_2020.csv')
sns.set(style='darkgrid')
ax = sns.countplot(x=nba.Pos, data=nba)
sns.pairplot(nba[["AST", "FG", "TRB"]])
plt.show()
correlation = nba[["AST", "FG", "TRB"]].corr()
sns.heatmap(correlation, annot=True)

from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = nba._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
labels