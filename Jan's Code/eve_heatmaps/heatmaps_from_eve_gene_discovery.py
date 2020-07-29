import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools as iter
import time


####dataset
####dataset
####dataset
models_raw = pd.read_csv("eve-genediscovery/EXPORT_eve_tp3_allStripes_geneDiscovery_ed2.csv")

#formating the "model" column into a list of gene names
models_column = list(models_raw["Model"])
for i in range(len(models_column)):
    models_column[i] = models_column[i].replace("'", "")
    models_column[i] = models_column[i][1:-1]
    models_column[i] = models_column[i].split(", ")
models_raw["Model"] = models_column

#converting Stripe columns to str() - for some reason its mixed type
models_raw["Stripe"] = models_raw["Stripe"].to_numpy(dtype=str)

#subseting by stripe
stripe_models = {}
for i in models_raw["Stripe"].unique():
    stripe_models[str(i)] = {}
    stripe_models[str(i)]["dataframe"] = pd.DataFrame(models_raw[models_raw["Stripe"] == i])

##creating list of individual unique genes
#because the model names are identical for each stripe, this can be done only once and copied over

genes = []
for i in list(stripe_models["3+7"]["dataframe"]["Model"]):
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
    models = list(stripe_models[i]["dataframe"]["Model"])
    MCC = list(stripe_models[i]["dataframe"]["MCC_median"])
    for j in gene_pairs:
        j = list(j)
        MCCs = []
        for x in range(len(models)):
            if all(u in models[x] for u in j):
                if str(j) in list(stripe_models[i]["pair_MCC_scores"].keys()):
                    if MCC[x] > stripe_models[i]["pair_MCC_scores"][str(j)]:
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
            test_pair = [j, x]
            for e in gene_pairs:
                e = list(e)
                if all(u in e for u in test_pair):
                    heatmap_row_dict[str(x)] = stripe_models[i]["pair_MCC_scores"][str(e)]

        heatmap_datasets[i]["raw"] += [heatmap_row_dict]
    heatmap_datasets[i]["dataframe"] = pd.DataFrame(heatmap_datasets[i]["raw"], index=genes)

#plotting heatmaps
gene_names = []
for i in genes:
    gene_names += [i[:-3]]
heatmaps = {}
for i in stripe_models.keys(): #stripe_models.keys()
    plt.figure(figsize=(12, 9))
    sns.set(font_scale=1)
    heatmaps[i] = sns.heatmap(heatmap_datasets[i]["dataframe"], linewidths=.5, xticklabels=gene_names, yticklabels=gene_names, cmap=plt.cm.OrRd, vmin=0.5, vmax=1, square=True)
    heatmaps[i].set_title(f"GeneDiscovery TP3 Eve stripe: {i}; MCC median")
    heatmaps[i] = heatmaps[i].get_figure()
    heatmaps[i].savefig(f"eve_heatmaps/second_ed/bdtnp_tp3_eve_stripe-{i}_heatmap_medianMCC.png")
    plt.close()
    time.sleep(0.2)