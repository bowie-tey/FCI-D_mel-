import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")
stripe_data = pd.read_csv("Jans_stripes_from_bdtnp.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

#subseting for timepoint 3
timepoint3_dataset = pd.DataFrame(dataset.iloc[:,dataset.columns.get_loc("x__3"):dataset.columns.get_loc("x__4")])
#adding bcdP quadratic
timepoint3_dataset["bcdP^2__3"] = np.array(timepoint3_dataset["bcdP__3"])**2

#adding combo-stripe identity columns
combo_stripe37 = []
combo_stripe46 = []
for i in list(stripe_data["all_stripes"]):
    if (i == 3) or (i == 7):
        combo_stripe37 += [1]
        combo_stripe46 += [0]
    elif (i == 4) or (i == 6):
        combo_stripe37 += [0]
        combo_stripe46 += [1]
    else:
        combo_stripe37 += [0]
        combo_stripe46 += [0]
stripe_data["stripe_3+7"] = combo_stripe37
stripe_data["stripe_4+6"] = combo_stripe46

#manually specifing the model genes based on gene discovery
stripe_model_genes = {"1": ["gtP__3", "Traf1__3", "prd__3", "Dfd__3"],
                      "2": ["bcdP__3", "bcdP^2__3", "hbP__3", "gtP__3", "KrP__3"],
                      "3+7": ["gtP__3", "kni__3", "tll__3", "hbP__3"],
                      "4+6": ["odd__3", "KrP__3", "gtP__3", "hbP__3"],
                      "5": ["kni__3", "hbP__3", "gtP__3", "KrP__3"]}

######################regression
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

models = {}
for i in stripe_model_genes.keys():
    models[i] = {}
for i in stripe_model_genes.keys():
    models[i]["eve_ex"] = timepoint3_dataset[stripe_model_genes[i]]
    models[i]["eve_in"] = stripe_data[f"stripe_{i}"]

    #balancing
    os = SMOTE(random_state=0)
    models[i]["eve_ex_train"], models[i]["eve_ex_test"], models[i]["eve_in_train"], models[i]["eve_in_test"] = \
        train_test_split(models[i]["eve_ex"], models[i]["eve_in"], test_size=0.3, random_state=0)
    models[i]["eve_ex_os"], models[i]["eve_in_os"] = os.fit_sample(models[i]["eve_ex_train"], models[i]["eve_in_train"])

    #modeling
    models[i]["model"] = LogisticRegression(C=1e10)
    models[i]["model"].fit(models[i]["eve_ex_os"], models[i]["eve_in_os"])

    #predicting
    models[i]["predict_proba_test"] = models[i]["model"].predict_proba(models[i]["eve_ex_test"])

    #determining the best proba threshold for each stripe
    threshold = []
    MCC = []
    for j in list(np.arange(0.60, 0.95, 0.01)):
        threshold += [j]
        predict_onoff = np.array([])
        for e in range(len(models[i]["predict_proba_test"])):
            if models[i]["predict_proba_test"][e, 1] >= j:
                predict_onoff = np.append(predict_onoff, True)
            elif models[i]["predict_proba_test"][e, 1] < j:
                predict_onoff = np.append(predict_onoff, False)
            else:
                print("error: predict proba -> onoff")
                break
        MCC += [sklearn.metrics.matthews_corrcoef(models[i]["eve_in_test"], predict_onoff)]
    MCC_threshold_analysis = pd.DataFrame(list(zip(threshold, MCC)), columns=["threshold", "MCC"])
    models[i]["max_threshold"] = MCC_threshold_analysis.iloc[MCC_threshold_analysis["MCC"].argmax()][0]

    models[i]["predict_onoff_test"] = np.array([])
    for e in range(len(models[i]["predict_proba_test"])):
        if models[i]["predict_proba_test"][e, 1] >= models[i]["max_threshold"]:
            models[i]["predict_onoff_test"] = np.append(models[i]["predict_onoff_test"], True)
        elif models[i]["predict_proba_test"][e, 1] < models[i]["max_threshold"]:
            models[i]["predict_onoff_test"] = np.append(models[i]["predict_onoff_test"], False)
        else:
            print("error: predict proba2 -> onoff")
            break
    models[i]["best_test_MCC"] = sklearn.metrics.matthews_corrcoef(models[i]["eve_in_test"], models[i]["predict_onoff_test"])

#ploting preparation
#testing models on the whole dataset

for i in models.keys():
    #predicting
    models[i]["predict_proba_whole_emb"] = models[i]["model"].predict_proba(models[i]["eve_ex"])
    models[i]["predict_onoff_whole_emb"] = np.array([])
    for e in range(len(models[i]["predict_proba_whole_emb"])):
        if models[i]["predict_proba_whole_emb"][e, 1] >= models[i]["max_threshold"]:
            models[i]["predict_onoff_whole_emb"] = np.append(models[i]["predict_onoff_whole_emb"], True)
        elif models[i]["predict_proba_whole_emb"][e, 1] < models[i]["max_threshold"]:
            models[i]["predict_onoff_whole_emb"] = np.append(models[i]["predict_onoff_whole_emb"], False)
        else:
            print("error: predict proba2 -> onoff")
            break
    models[i]["best_whole_emb_MCC"] = sklearn.metrics.matthews_corrcoef(models[i]["eve_in"],
                                                                   models[i]["predict_onoff_whole_emb"])


    models[i]["match_identity"] = np.array([])
    for e in range(len(models[i]["predict_onoff_whole_emb"])):
        # counting the number of false pos, false neg, and true matches
        if models[i]["predict_onoff_whole_emb"][e] != models[i]["eve_in"][e] and models[i]["eve_in"][e] == True:
            models[i]["match_identity"] = np.append(models[i]["match_identity"], "false negative")
        elif models[i]["predict_onoff_whole_emb"][e] != models[i]["eve_in"][e] and models[i]["eve_in"][e] == False:
            models[i]["match_identity"] = np.append(models[i]["match_identity"], "false positive")
        elif models[i]["predict_onoff_whole_emb"][e] == models[i]["eve_in"][e] and models[i]["eve_in"][e] == True:
            models[i]["match_identity"] = np.append(models[i]["match_identity"], "match positive")
        elif models[i]["predict_onoff_whole_emb"][e] == models[i]["eve_in"][e] and models[i]["eve_in"][e] == False:
            models[i]["match_identity"] = np.append(models[i]["match_identity"], "match negative")
        else:
            print(error)
            break

    models[i]["colors"] = np.array([])
    for e in range(len(models[i]["match_identity"])):
        #color coding match identity for ploting
        if models[i]["match_identity"][e] == "false positive":
            models[i]["colors"] = np.append(models[i]["colors"], "#f130ff")
        elif models[i]["match_identity"][e] == "false negative":
            models[i]["colors"] = np.append(models[i]["colors"], "#ff1900")
        elif models[i]["match_identity"][e] == "match positive":
            models[i]["colors"] = np.append(models[i]["colors"], "#0062ff")
        elif models[i]["match_identity"][e] == "match negative":
            models[i]["colors"] = np.append(models[i]["colors"], "#fafafa")

validation_figures ={}
for i in models.keys():
    #saving the number of false positives, negatives, true positives and negatives
    models[i]["fp"] = pd.DataFrame(np.unique(models[i]["match_identity"], return_counts=True))[1][1]
    models[i]["fn"] = pd.DataFrame(np.unique(models[i]["match_identity"], return_counts=True))[0][1]
    models[i]["tp"] = pd.DataFrame(np.unique(models[i]["eve_in"], return_counts=True))[1][1]
    models[i]["tn"] = pd.DataFrame(np.unique(models[i]["eve_in"], return_counts=True))[0][1]

    #ploting
    validation_figures[i] = go.Figure()
    validation_figures[i].add_trace(
        go.Scatter3d(
            x=timepoint3_dataset["x__3"],
            y=timepoint3_dataset["y__3"],
            z=timepoint3_dataset["z__3"],
            mode='markers',
            marker=dict(color=models[i]["colors"],
                        size=8
                        )
        )
    )
    validation_figures[i].update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                                                height=700,
                                                width=1200,
                                                title="Stripe: {i}; Genes: {genes}; Coefs: {coefs}".format(i = i,
                                                                                             genes = stripe_model_genes[i],
                                                                                             coefs = np.round(models[i]["model"].coef_[0], 2)),
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
                                                        text="<b>MCC:</b> {MCC:.5f}<br>"
                                                             "<b>Threshold:</b> {THR:.2f}<br>"
                                                             "<b>True Positive:</b> {TP}<br>"
                                                             "<b>False positive:</b> {FP}<br>"
                                                             "<b>False negative:</b> {FN}".format(
                                                            MCC=models[i]["best_whole_emb_MCC"],
                                                            THR=models[i]["max_threshold"],
                                                            FP=models[i]["fp"],
                                                            FN=models[i]["fn"],
                                                            TP=models[i]["tp"]),
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
                                                        y=0.55,
                                                        bordercolor='black',
                                                        borderwidth=1)]
                                                ##ANNOTATIONS
                                                ##ANNOTATIONS
                                                )
    validation_figures[i].show()

#making an interactive plot with all stripes merged into one and togless to choose between stripes as well
#making a complete color column - all models in one plot
complete_colors = ["#fafafa"] * len(models["1"]["colors"])
for i in models.keys():
    for e in range(len(models["1"]["colors"])):
        if models[i]["colors"][e] != complete_colors[e] and complete_colors[e] == "#fafafa":
            complete_colors[e] = models[i]["colors"][e]

complete_figure = go.Figure()
complete_figure.add_trace(
        go.Scatter3d(
            x=timepoint3_dataset["x__3"],
            y=timepoint3_dataset["y__3"],
            z=timepoint3_dataset["z__3"],
            mode='markers',
            visible= True,
            marker=dict(color=complete_colors,
                        size=8
                        )
        )
)
for i in models.keys():
    complete_figure.add_trace(
        go.Scatter3d(
            x=timepoint3_dataset["x__3"],
            y=timepoint3_dataset["y__3"],
            z=timepoint3_dataset["z__3"],
            mode='markers',
            visible= False,
            marker=dict(color=models[i]["colors"],
                        size=8
                        )
        )
    )
complete_figure.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                            height=700,
                            width=1200,
                            scene=dict(
                      xaxis=dict(range=[0, 440]),
                      yaxis=dict(range=[0, 160]),
                      zaxis=dict(range=[0, 160]),
                      xaxis_title="Anterior-Posterior Axis (um)",
                      zaxis_title="Dorso-Ventral Axis (um)",
                      yaxis_title="Left-Right Axis (um)")
                            )

#creating annotations
anotations = {}
for i in models.keys():
    anotations[i] = [
        dict(text="<b>MCC:</b> {MCC:.5f}<br>"
                  "<b>Threshold:</b> {THR:.2f}<br>"
                  "<b>True Positive:</b> {TP}<br>"
                  "<b>False positive:</b> {FP}<br>"
                  "<b>False negative:</b> {FN}".format(MCC=models[i]["best_whole_emb_MCC"],
                                                       THR=models[i]["max_threshold"],
                                                       FP=models[i]["fp"],
                                                       FN=models[i]["fn"],
                                                       TP=models[i]["tp"]),
            align='left',
            showarrow=False,
            xref='paper',
            yref='paper',
            x=0.98,
            y=0.8,
            bordercolor='black',
            borderwidth=1),

        dict(text="<b><i>Colors:</i></b><br>"
                  "<b>Positive:</b> blue<br>"
                  "<b>False positive:</b> purple<br>"
                  "<b>False negative:</b> red",
             align='left',
             showarrow=False,
             xref='paper',
             yref='paper',
             x=0.98,
             y=0.55,
             bordercolor='black',
             borderwidth=1)
    ]
anotations["all"] = [
    dict(text="<b><i>Colors:</i></b><br>"
                  "<b>Positive:</b> blue<br>"
                  "<b>False positive:</b> purple<br>"
                  "<b>False negative:</b> red",
             align='left',
             showarrow=False,
             xref='paper',
             yref='paper',
             x=0.98,
             y=0.55,
             bordercolor='black',
             borderwidth=1)
]

#adding the buttons allowing for selection of displayed stripes
complete_figure.update_layout(
            updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=0,
            buttons=list([
                dict(label="all",
                     method="update",
                     args=[{"visible": [True, False, False, False, False, False]},
                           {"title": "",
                            "annotations": anotations["all"]}]),
                dict(label="1",
                     method="update",
                     args=[{"visible": [False, True, False, False, False, False]},
                           {"title": "Stripe: {i}; Genes: {genes}; Coefs: {coefs}".format(i = "1",
                                                                                          genes = stripe_model_genes["1"],
                                                                                          coefs = np.round(models["1"]["model"].coef_[0], 2)),
                            "annotations": anotations["1"]}]),
                dict(label="2",
                     method="update",
                     args=[{"visible": [False, False, True, False, False, False]},
                           {"title": "Stripe: {i}; Genes: {genes}; Coefs: {coefs}".format(i = "2",
                                                                                          genes = stripe_model_genes["2"],
                                                                                          coefs = np.round(models["2"]["model"].coef_[0], 2)),
                            "annotations": anotations["2"]}]),
                dict(label="3+7",
                     method="update",
                     args=[{"visible": [False, False, False, True, False, False]},
                           {"title": "Stripe: {i}; Genes: {genes}; Coefs: {coefs}".format(i = "3+7",
                                                                                          genes = stripe_model_genes["3+7"],
                                                                                          coefs = np.round(models["3+7"]["model"].coef_[0], 2)),
                            "annotations": anotations["3+7"]}]),
                dict(label="4+6",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, False]},
                           {"title": "Stripe: {i}; Genes: {genes}; Coefs: {coefs}".format(i = "4+6",
                                                                                          genes = stripe_model_genes["4+6"],
                                                                                          coefs = np.round(models["4+6"]["model"].coef_[0], 2)),
                            "annotations": anotations["4+6"]}]),
                dict(label="5",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, True]},
                           {"title": "Stripe: {i}; Genes: {genes}; Coefs: {coefs}".format(i = "5",
                                                                                          genes = stripe_model_genes["5"],
                                                                                          coefs = np.round(models["5"]["model"].coef_[0], 2)),
                            "annotations": anotations["5"]}]),
            ]),
        )
    ])

complete_figure.show()