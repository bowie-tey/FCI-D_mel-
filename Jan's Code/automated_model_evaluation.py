import numpy as np
import pandas as pd

####dataset
####dataset
####dataset

models_raw = pd.read_csv("automated_model_search.csv")

#formating the "model" column into a list of gene names
models_column = list(models_raw["model"])
for i in range(len(models_column)):
    models_column[i] = models_column[i].replace("'", "")
    models_column[i] = models_column[i][1:-1]
    models_column[i] = models_column[i].split(", ")
models_raw["model"] = models_column

#subseting by stripe
stripe_models = {}
for i in models_raw["stripe"].unique():
    stripe_models[str(i)] = pd.DataFrame(models_raw[models_raw["stripe"] == i])

####scoring genes
####scoring genes
####scoring genes
####scoring genes
scoring = {} #scoring is separate for every stipe

##creating list of individual unique genes
#because the model names are identical for each stripe, this can be done only once and copied over

genes = []
for i in list(stripe_models["1"]["model"]):
    for x in i:
        if not x in genes:
            genes += [x]