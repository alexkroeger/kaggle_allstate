import pandas as pd
import numpy as np

base = "/home/noah/Documents/kaggle_allstate/"

train = pd.read_csv(base + "data/train.csv", index_col = 0)

two = df[[c for c in train.columns if len(train[c].unique()) == 2]]

pairs = [(x, y) for x in two.columns for y in two.columns if x != y]

# add loss column 
two["loss"] = df["loss"].copy()

classifiers = [two.groupby([x, y])["loss"].mean().to_dict() for x, y in pairs]

def classify(row):
    preds = [d[(row[x], row[y])] for (x, y), d in zip(pairs, classifiers)]
    return np.mean(preds)

two["pred"] = two.apply(classify, axis = 1)


