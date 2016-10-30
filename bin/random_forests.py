import pandas as pd
import numpy as np
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

base = "/home/noah/Documents/kaggle_allstate/"
x = 1000

#### ingest and pre-process data
# load train and test data
train = pd.read_csv(base + "data/train.csv", index_col = 0)
test = pd.read_csv(base + "data/test.csv", index_col = 0)

# get class label encoders for each cat variable across train and test set
cat_class_labs = {c: np.concatenate((train[c].unique(), test[c].unique()))
                  for c in train.columns if c[:3] == "cat"}
encoders = {k: LabelEncoder().fit(v) for k, v in cat_class_labs.items()}

# transform each column
for label in train.columns:
    if label[:3] == "cat":
        train[label] = encoders[label].transform(train[label])
        test[label] = encoders[label].transform(test[label])


#### define model, cv parameters, train the model
# make model object
rfr = RandomForestRegressor(criterion = "mae", random_state = 1,
                            n_jobs = -1)

# cross validation parameters
n_steps = 5
start, step = 2, 4
max_features_range = list(range(start, start+(step*n_steps), step))
start, step = 4, 4
n_estimators_range = list(range(start, start+(step*n_steps), step))

param_grid = {'max_features': max_features_range,
              'n_estimators': n_estimators_range}

# train cv'd model
cv = GridSearchCV(rfr, param_grid = param_grid, refit = True, cv = 5,
                  verbose = 2, scoring = "neg_mean_absolute_error",
                  n_jobs = 4)
cv.fit(train.drop("loss", axis = 1).head(x), train["loss"].head(x))

# classify the test set with the best cv'd model
test["loss"] = cv.predict(test)
test[["loss"]].to_csv(base + "predictions/rf.csv")


#### plot CV loss
fig = plt.figure()
ax = fig.gca(projection = '3d')
x_3d = cv.cv_results_["param_max_features"].data.reshape((n_steps,)*2)
y_3d = cv.cv_results_["param_n_estimators"].data.reshape((n_steps,)*2)
z_3d = cv.cv_results_["mean_test_score"].reshape((n_steps,)*2)*-1
surf = ax.plot_surface(x_3d, y_3d, z_3d, cmap = cm.summer,
                       rstride = 1, cstride = 1)
ax.set_zlim(0, z_3d.max().max()*1.05)
fig.colorbar(surf, shrink = 0.5, aspect = 5)
plt.savefig(base + "output/rf_cv_plot.png")

