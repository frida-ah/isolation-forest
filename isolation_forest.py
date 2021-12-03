# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest


# %%
# California housing prices
pdf = pd.read_csv(
    "https://raw.githubusercontent.com/flyandlure/datasets/master/housing.csv"
)
pdf.head()


# %%
sns.violinplot(x=pdf["median_house_value"])


# %%
# sns.boxplot(data=pdf["median_house_value"])


# %%
# Initiate the model
model = IsolationForest(
    n_estimators=50, max_samples="auto", contamination=float(0.1), max_features=1.0
)

# n_estimators = # base estimators in the ensemble (default = 100)
# max_samples = # samples to draw from X to train each base estimator (default = "auto")
# contamination = Amount of contamination of the data set, i.e. the proportion of outliers in the data set.
#                 float:(0, 0.5]
# max_features = # features to draw from X to train each base estimator. (default = 1.0)


model.fit(pdf[["median_house_value"]])


# %%
# help(IsolationForest)


# %%
# Average anomaly score of X
pdf["scores"] = model.decision_function(pdf[["median_house_value"]])

# Perform fit on X and returns labels for X. Returns -1 for outliers and 1 for inliers.
pdf["anomaly"] = model.predict(pdf[["median_house_value"]])

anomaly = pdf.loc[pdf["anomaly"] == -1]
anomaly_index = list(anomaly.index)

pdf[["scores", "anomaly"]].head(5)
# 1 = normal
# -1 = anomaly

# %% [markdown]
# Anomaly score = mean anomaly scores of trees in the forest.
#
# The measure of normality of an observation given a tree is the depth of the leaf containing this observation, which is equivalent to the number of splittings required to isolate this point.

# %%
# model evaluation
# define a threshold for outlier limit
threshold = 400000

total_counter = pdf["median_house_value"].count()

outliers_counter = len(pdf[pdf["median_house_value"] > threshold])

print(
    f"Houses in dataset = {total_counter}\n"
    f"Houses with price higher than {threshold}€ = {outliers_counter}"
)

print("Accuracy percentage:", 100 * list(pdf["anomaly"]).count(-1) / (outliers_counter))


# %%
