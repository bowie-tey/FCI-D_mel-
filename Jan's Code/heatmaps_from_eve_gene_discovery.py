import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as iter
import time


####dataset
####dataset
####dataset
models_raw = pd.read_csv("automated_model_search.csv")
models_raw2 = pd.read_csv("automated_model_search2.csv")
models_raw_all = pd.concat([models_raw, models_raw2])

#formating the "model" column into a list of gene names
models_column = list(models_raw_all["model"])
for i in range(len(models_column)):
    models_column[i] = models_column[i].replace("'", "")
    models_column[i] = models_column[i][1:-1]
    models_column[i] = models_column[i].split(", ")
models_raw_all["model"] = models_column


#subseting by stripe
stripe_models = {}
for i in models_raw_all["stripe"].unique():
    stripe_models[str(i)] = {}
    stripe_models[str(i)]["dataframe"] = pd.DataFrame(models_raw_all[models_raw_all["stripe"] == i])

##creating list of individual unique genes
#because the model names are identical for each stripe, this can be done only once and copied over

genes = []
for i in list(stripe_models["3+7"]["dataframe"]["model"]):
    for x in i:
        if x not in genes:
            genes += [x]

#finding the best model for each pair of genes
#finding the best model for each pair of genes

gene_pairs = list(iter.combinations(genes, 2))
#selfcombination
self_combo = []
for i in genes:
    self_combo += [[i]*2]
gene_pairs += self_combo

for i in stripe_models.keys():
    print("start", i)
    stripe_models[i]["pair_MCC_scores"] = {}
    models = list(stripe_models[i]["dataframe"]["model"])
    MCC = list(stripe_models[i]["dataframe"]["MCC"])
    for j in gene_pairs:
        j = list(j)
        for x in range(len(models)):
            if j[0] in models[x] and j[1] in models[x]:
                if j in list(stripe_models[i]["pair_MCC_scores"].keys()):
                    if MMC[x] > stripe_models[i]["pair_MCC_scores"][j]:
                        stripe_models[i]["pair_MCC_scores"][str(j)] = MCC[x]
                    else:
                        pass
                else:
                    stripe_models[i]["pair_MCC_scores"][str(j)] = MCC[x]
    print("end", i)

#generating heatmap matrices
heatmap_datasets= {}
for i in stripe_models.keys():
    heatmap_datasets[i] = {}
    heatmap_datasets[i]["raw"] = []
    for j in genes: # row
        heatmap_row_dict = {}
        for x in genes: #colums
            for e in gene_pairs:
                e = list(e)
                if (j in e[0] and x in e[1]) or (x in e[0] and j in e[1]):
                    heatmap_row_dict[str(x)] = stripe_models[i]["pair_MCC_scores"][str(e)]

        heatmap_datasets[i]["raw"] += [heatmap_row_dict]
    heatmap_datasets[i]["numpy"] = pd.DataFrame(heatmap_datasets[i]["raw"], index=genes).to_numpy()

#plotting heatmaps
gene_names = []
for i in genes:
    gene_names += [i[:-3]]
heatmaps = {}
for i in stripe_models.keys():
    plt.figure(figsize=(12, 9))
    sns.set(font_scale=1)
    heatmaps[i] = sns.heatmap(heatmap_datasets[i]["numpy"], linewidths=.5, xticklabels=gene_names, yticklabels=gene_names, cmap=plt.cm.OrRd, square=True)
    heatmaps[i].set_title(f"GeneDiscovery TP3 Eve stripe: {i}")
    heatmaps[i] = heatmaps[i].get_figure()
    heatmaps[i].savefig(f"heatmaps/bdtnp_tp3_eve_stripe-{i}_heatmap.png")
    plt.close()
    time.sleep(0.2)
