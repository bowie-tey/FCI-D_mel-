import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly
import plotly.graph_objects as go

###########FUNCTIONS
###########FUNCTIONS
def proba_to_onoff(predict_proba, threshold):
    predict_proba_vector = predict_proba[:,1]

    predict_onoff = ["NaN"] * len(predict_proba_vector)
    for i in range(len(predict_proba_vector)):
        if predict_proba_vector[i] > threshold:
            predict_onoff[i] = 1
        elif predict_proba_vector[i] <= threshold:
            predict_onoff[i] = 0
        else:
            raise Exception(f"proba_to_onoff - ERROR - script out of bounds; %{type(predict_proba_vector[i])}%")

    if "NaN" in predict_onoff:
        raise Exception("proba_to_onoff - ERROR - predict_onoff contains NaN")

    return predict_onoff
def match_mismatch(predict_onoff, test):
    match_mismatch = np.array([])

    # counting the number of false pos, false neg, and true matches
    for predict, tes in zip(predict_onoff, test):
        if predict != tes and tes == True:
            match_mismatch = np.append(match_mismatch, "false negative")
        elif predict != tes and tes == False:
            match_mismatch = np.append(match_mismatch, "false positive")
        elif predict == tes and tes == True:
            match_mismatch = np.append(match_mismatch, "match positive")
        elif predict == tes and tes == False:
            match_mismatch = np.append(match_mismatch, "match negative")
        else:
            raise Exception("match_mismatch - ERROR - script out of bounds")

    return match_mismatch
###########FUNCTIONS
###########FUNCTIONS

#importing dataset
gene_discovery_dataset = pd.read_csv("eve-genediscovery/EXPORT_eve_tp3_allStripes_geneDiscovery_ed2.csv")

#formating the "model" column into a list of gene names
models_column = list(gene_discovery_dataset["Model"])
for i in range(len(models_column)):
    models_column[i] = models_column[i].replace("'", "")
    models_column[i] = models_column[i][1:-1]
    models_column[i] = models_column[i].split(", ")
gene_discovery_dataset["Model"] = models_column

#subseting by stripe
stripe_models = {}
for i in gene_discovery_dataset["Stripe"].unique():
    stripe_models[str(i)] = gene_discovery_dataset[gene_discovery_dataset["Stripe"] == i]

#print top 20 models for 4+6 stripe
top_stripe46 = stripe_models["4+6"].sort_values(by=["MCC_median"], ascending=False).iloc[:20,:]

#making a list of all models to count how many times a certain gene occurs in the top20 models
top_stripe46_model_list = []
for i in list(top_stripe46["Model"]):
    top_stripe46_model_list += i

#list of genes and their occurences in the top20 models
top_stripe46_gene_list = pd.DataFrame(np.unique(np.array(top_stripe46_model_list), return_counts=True)).T.sort_values(by=[1], ascending=False)
top_stripe46_gene_list.columns = ["Gene", "Count"]
#based on this analysis KrP, hbP, gtP, and kni have been identified as the top genes
#this finding aligns with the performed heatmap analysis

###TESTING THE MODEL
###TESTING THE MODEL
###TESTING THE MODEL

#importing the dataset
dataset = pd.read_csv("D_mel_atlas.csv")
stripe_identity = pd.read_csv("Jans_stripes_from_bdtnp.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

#adding combo-stripe identity columns
combo_stripe37 = []
combo_stripe46 = []
for i in list(stripe_identity["all_stripes"]):
    if (i == 3) or (i == 7):
        combo_stripe37 += [1]
        combo_stripe46 += [0]
    elif (i == 4) or (i == 6):
        combo_stripe37 += [0]
        combo_stripe46 += [1]
    else:
        combo_stripe37 += [0]
        combo_stripe46 += [0]
stripe_identity["stripe_3+7"] = combo_stripe37
stripe_identity["stripe_4+6"] = combo_stripe46

#selecting relevant timepoint 3 data
model_dataset = pd.concat([dataset[["x__3", "y__3", "z__3", "KrP__3", "hbP__3", "gtP__3", "kni__3"]], stripe_identity["stripe_4+6"]], axis=1)

##REGRESSION
#balancing
eve_ex = model_dataset[["KrP__3", "hbP__3", "gtP__3", "kni__3"]].to_numpy()
eve_in = model_dataset["stripe_4+6"].to_numpy()

os = SMOTE(random_state=0)
eve_ex_train, eve_ex_test, eve_in_train, eve_in_test = train_test_split(eve_ex, eve_in, test_size=0.3, random_state=0)
os_data_eve_ex,os_data_eve_in=os.fit_sample(eve_ex_train, eve_in_train)

#model
model = LogisticRegression(C=1e10)
model.fit(os_data_eve_ex, os_data_eve_in)

#MCC analysis
predict_proba = model.predict_proba(eve_ex)
predict_onoff = proba_to_onoff(predict_proba, 0.75)
MCC = sklearn.metrics.matthews_corrcoef(eve_in, predict_onoff)
print(MCC)

match_identity = match_mismatch(predict_onoff, eve_in)

#making color array for plotting below
colors = np.array([])
for i in match_identity:
    if i == "false positive":
        colors = np.append(colors, "#f130ff") #purple
    elif i == "false negative":
        colors = np.append(colors, "#ff1900") #red
    elif i == "match positive":
        colors = np.append(colors, "#0062ff") #blue
    elif i == "match negative":
        colors = np.append(colors, "#fafafa") #white

#saving the number of false positives, negatives, true positives and negatives
fp = (match_identity == "false positive").sum()
fn = (match_identity == "false negative").sum()
tp = (model_dataset["stripe_4+6"] == 1).sum()
tn = (model_dataset["stripe_4+6"] == 0).sum()

###PLOTING
###PLOTING
###PLOTING

plot = go.Figure()
plot.add_trace(
    go.Scatter3d(
        x = model_dataset["x__3"],
        y = model_dataset["y__3"],
        z = model_dataset["z__3"],
        mode='markers',
            marker=dict(color=colors,
                        size=8
                        )
    )
)

plot.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                      height=700,
                      width=1200,
                      title = "KrP__3, hbP__3, gtP__3, kni__3; EVE 4+6",
                      scene=dict(
                          xaxis=dict(range=[0, 440]),
                          yaxis=dict(range=[0, 160]),
                          zaxis=dict(range=[0, 160]),
                          xaxis_title="Anterior-Posterior Axis (um)",
                          zaxis_title="Dorso-Ventral Axis (um)",
                          yaxis_title="Left-Right Axis (um)"),
                      ##ANNOTATIONS
                      ##ANNOTATIONS
                      annotations=[
                          dict(
                              text=f"<b>MCC:</b> {MCC:.5f}<br>"
                                   f"<b>True Positive:</b> {tp}<br>"
                                   f"<b>False positive:</b> {fp}<br>"
                                   f"<b>False negative:</b> {fn}",
                              align='left',
                              showarrow=False,
                              xref='paper',
                              yref='paper',
                              x=0.98,
                              y=0.8,
                              bordercolor='black',
                              borderwidth=1),
                          dict(
                              text="<b><i>Colors:</i></b><br>"
                                   "<b>Positive:</b> blue<br>"
                                   "<b>False positive:</b> purple<br>"
                                   "<b>False negative:</b> red",
                              align='left',
                              showarrow=False,
                              xref='paper',
                              yref='paper',
                              x=0.98,
                              y=0.6,
                              bordercolor='black',
                              borderwidth=1)
                      ##ANNOTATIONS
                      ##ANNOTATIONS
                          ]
)
plot.show()