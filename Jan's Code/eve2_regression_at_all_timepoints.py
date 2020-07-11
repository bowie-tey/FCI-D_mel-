import pandas as pd
import numpy as np

import itertools


###TIMEPOINTS TO ANALYSE
timepoints = 3
###TIMEPOINTS TO ANALYSE

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

regression_dataset_dict = {}
for x in range(timepoints):
    regression_dataset_dict[f"tp{x+1}"] = pd.DataFrame(dataset[[f"x__{x+1}", f"y__{x+1}", f"z__{x+1}", f"eve__{x+1}", f"bcdP__{x+1}", f"hbP__{x+1}", f"gtP__{x+1}", f"KrP__{x+1}"]])
    regression_dataset_dict[f"tp{x+1}"][f"bcdP^2__{x+1}"] = np.array(regression_dataset_dict[f"tp{x+1}"][f"bcdP__{x+1}"])**2

######################stripe 2 isolation

#THRESHOLD
eve_threshold = 0.165
#THRESHOLD

position_determinant_dict = {}
for x in range(timepoints):
    position_determinant_dict[f"tp{x+1}"] = ((dataset[f"x__{x+1}"] >= 145) & (dataset[f"x__{x+1}"] <= 176) & (dataset[f"z__{x+1}"] >= 70) & (dataset[f"eve__{x+1}"] >= eve_threshold)) | \
                                                                        ((dataset[f"x__{x+1}"] >= 131) & (dataset[f"x__{x+1}"] <= 170) & (dataset[f"z__{x+1}"] < 70) & (dataset[f"eve__{x+1}"] >= eve_threshold))


#adding a "on/off" column to regression_dataset (converts boolean to int)
for x in range(timepoints):
    regression_dataset_dict[f"tp{x+1}"]["eve_onoff"] = np.array(position_determinant_dict[f"tp{x+1}"])*1

######################regression
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#balancing
eve_ex_dict = {"test": {}, "train": {}}
eve_in_dict = {"test": {}, "train": {}}
os_dict = {"eve_in": {}, "eve_ex": {}}

for x in range(timepoints):
    eve_ex_dict[f"tp{x + 1}"] = pd.DataFrame.to_numpy(regression_dataset_dict[f"tp{x+1}"][[f"bcdP__{x+1}", f"bcdP^2__{x+1}", f"hbP__{x+1}", f"gtP__{x+1}", f"KrP__{x+1}"]])
    eve_in_dict[f"tp{x + 1}"] = pd.DataFrame.to_numpy(regression_dataset_dict[f"tp{x+1}"]["eve_onoff"])

    os = SMOTE(random_state=0)

    eve_ex_dict["train"][f"tp{x+1}"], \
    eve_ex_dict["test"][f"tp{x+1}"], \
    eve_in_dict["train"][f"tp{x + 1}"], \
    eve_in_dict["test"][f"tp{x + 1}"] = \
    train_test_split(eve_ex_dict[f"tp{x + 1}"],
                         eve_in_dict[f"tp{x + 1}"],
                         test_size=0.3,
                         random_state=0)
    os_dict["eve_ex"][f"tp{x+1}"], os_dict["eve_in"][f"tp{x+1}"] = os.fit_sample(eve_ex_dict["train"][f"tp{x+1}"], eve_in_dict["train"][f"tp{x + 1}"])

#model
logreg_models_dict = {}

for x in range(timepoints):
    logreg_models_dict[f"tp{x+1}"] = LogisticRegression()
    logreg_models_dict[f"tp{x + 1}"].fit(os_dict["eve_ex"][f"tp{x+1}"], os_dict["eve_in"][f"tp{x+1}"])
    print(f"timepoint: {x + 1}", logreg_models_dict[f"tp{x + 1}"].coef_)
##coef: ["bcdP__", "hbP__", "gtP__", "KrP__"]
#timepoint: 1[[-19.36660166  17.26118287 - 13.03516797 - 3.90388066]]
#timepoint: 2[[-16.46772599  13.86282708 - 11.20048436 - 4.28701761]]
#timepoint: 3[[-20.30643861  16.43566713 - 10.54650913 - 2.99629309]]

#verifying the validity of the model /w MCC
eve_predict_dict = {"proba": {}, "onoff": {}}
MCC_predict_dict = {"score": {}, "fp": {}, "tp": {}, "fn": {}, "tn": {}, "match_identity": {}}

##PREDICT PROBA TRESHOLD
predict_proba_threshold = 0.81
##PREDICT PROBA TRESHOLD

for x in range(timepoints):
    eve_predict_dict["onoff"][f"tp{x+1}"] = np.array([])
    MCC_predict_dict["match_identity"][f"tp{x + 1}"] = np.array([])

    #predicting eve expression in the test dataset
    eve_predict_dict["proba"][f"tp{x + 1}"] = logreg_models_dict[f"tp{x + 1}"].predict_proba(eve_ex_dict["test"][f"tp{x + 1}"])

    # determining whether eve is on or of based on the predicted probability
    for i in range(len(eve_predict_dict["proba"][f"tp{x+1}"])):
        if eve_predict_dict["proba"][f"tp{x+1}"][i,1] >= predict_proba_threshold:
            eve_predict_dict["onoff"][f"tp{x+1}"] = np.append(eve_predict_dict["onoff"][f"tp{x+1}"], True)
        elif eve_predict_dict["proba"][f"tp{x+1}"][i,1] < predict_proba_threshold:
            eve_predict_dict["onoff"][f"tp{x+1}"] = np.append(eve_predict_dict["onoff"][f"tp{x+1}"], False)
        else:
            print("error")
            break

        # counting the number of false pos, false neg, and true matches
        if eve_predict_dict["onoff"][f"tp{x+1}"][i] != eve_in_dict["test"][f"tp{x+1}"][i] and eve_in_dict["test"][f"tp{x+1}"][i] == True:
            MCC_predict_dict["match_identity"][f"tp{x+1}"] = np.append(MCC_predict_dict["match_identity"][f"tp{x+1}"], "false negative")
        elif eve_predict_dict["onoff"][f"tp{x+1}"][i] != eve_in_dict["test"][f"tp{x+1}"][i] and eve_in_dict["test"][f"tp{x+1}"][i] == False:
            MCC_predict_dict["match_identity"][f"tp{x+1}"] = np.append(MCC_predict_dict["match_identity"][f"tp{x+1}"], "false positive")
        elif eve_predict_dict["onoff"][f"tp{x+1}"][i] == eve_in_dict["test"][f"tp{x+1}"][i]:
            MCC_predict_dict["match_identity"][f"tp{x+1}"] = np.append(MCC_predict_dict["match_identity"][f"tp{x+1}"], "match")
        else:
            print(error)
            break

    MCC_predict_dict["score"][f"tp{x+1}"] = sklearn.metrics.matthews_corrcoef(eve_in_dict["test"][f"tp{x+1}"], eve_predict_dict["onoff"][f"tp{x+1}"])
    MCC_predict_dict["fp"][f"tp{x+1}"] = pd.DataFrame(np.unique(MCC_predict_dict["match_identity"][f"tp{x+1}"], return_counts = True))[1][1]
    MCC_predict_dict["fn"][f"tp{x+1}"] = pd.DataFrame(np.unique(MCC_predict_dict["match_identity"][f"tp{x+1}"], return_counts=True))[0][1]
    MCC_predict_dict["tp"][f"tp{x+1}"] = pd.DataFrame(np.unique(eve_in_dict["test"][f"tp{x+1}"], return_counts=True))[1][1]
    MCC_predict_dict["tn"][f"tp{x+1}"] = pd.DataFrame(np.unique(eve_in_dict["test"][f"tp{x+1}"], return_counts=True))[0][1]

######################embryo modelling
######################embryo modelling
######################embryo modelling
######################embryo modelling
######################embryo modelling
######################embryo modelling
######################embryo modelling
######################embryo modelling

import plotly
import plotly.graph_objects as go

embryo_eve_ex_dict = {}
embryo_eve_in_dict = {}

embryo_eve_predict_dict = {"proba": {}, "onoff": {}}
embryo_MCC_predict_dict = {"score": {}, "fp": {}, "tp": {}, "fn": {}, "tn": {}, "match_identity": {}}

embryo_fig_dict = {"colors": {}}

for x in range(timepoints):
    embryo_eve_ex_dict[f"tp{x+1}"] = pd.DataFrame.to_numpy(regression_dataset_dict[f"tp{x+1}"][[f"bcdP__{x+1}", f"bcdP^2__{x+1}", f"hbP__{x+1}", f"gtP__{x+1}", f"KrP__{x+1}"]])
    embryo_eve_in_dict[f"tp{x+1}"] = pd.DataFrame.to_numpy(regression_dataset_dict[f"tp{x+1}"]["eve_onoff"])

    embryo_eve_predict_dict["onoff"][f"tp{x + 1}"] = np.array([])
    embryo_MCC_predict_dict["match_identity"][f"tp{x + 1}"] = np.array([])

    embryo_fig_dict["colors"][f"tp{x + 1}"] = np.array([])

    #predicting eve expression in individual cells
    embryo_eve_predict_dict["proba"][f"tp{x + 1}"] = logreg_models_dict[f"tp{x + 1}"].predict_proba(embryo_eve_ex_dict[f"tp{x + 1}"])

    #determining whether eve is on or of based on the predicted probability
    for i in range(len(embryo_eve_predict_dict["proba"][f"tp{x + 1}"])):
        if embryo_eve_predict_dict["proba"][f"tp{x+1}"][i,1] >= predict_proba_threshold:
            embryo_eve_predict_dict["onoff"][f"tp{x+1}"] = np.append(embryo_eve_predict_dict["onoff"][f"tp{x+1}"], True)
        elif embryo_eve_predict_dict["proba"][f"tp{x+1}"][i,1] < predict_proba_threshold:
            embryo_eve_predict_dict["onoff"][f"tp{x+1}"] = np.append(embryo_eve_predict_dict["onoff"][f"tp{x+1}"], False)
        else:
            print("error")
            break

        # counting the number of false pos, false neg, and true matches
        if embryo_eve_predict_dict["onoff"][f"tp{x+1}"][i] != embryo_eve_in_dict[f"tp{x+1}"][i] and embryo_eve_in_dict[f"tp{x+1}"][i] == True:
            embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"] = np.append(embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"], "false negative")
        elif embryo_eve_predict_dict["onoff"][f"tp{x+1}"][i] != embryo_eve_in_dict[f"tp{x+1}"][i] and embryo_eve_in_dict[f"tp{x+1}"][i] == False:
            embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"] = np.append(embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"], "false positive")
        elif embryo_eve_predict_dict["onoff"][f"tp{x+1}"][i] == embryo_eve_in_dict[f"tp{x+1}"][i] and embryo_eve_in_dict[f"tp{x+1}"][i] == True:
            embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"] = np.append(embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"], "match positive")
        elif embryo_eve_predict_dict["onoff"][f"tp{x+1}"][i] == embryo_eve_in_dict[f"tp{x+1}"][i] and embryo_eve_in_dict[f"tp{x+1}"][i] == False:
            embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"] = np.append(embryo_MCC_predict_dict["match_identity"][f"tp{x+1}"], "match negative")
        else:
            print(error)
            break

        #making color array for plotting below
        if embryo_MCC_predict_dict["match_identity"][f"tp{x + 1}"][i] == "false positive":
            embryo_fig_dict["colors"][f"tp{x + 1}"] = np.append(embryo_fig_dict["colors"][f"tp{x + 1}"], "#f130ff")
        elif embryo_MCC_predict_dict["match_identity"][f"tp{x + 1}"][i] == "false negative":
            embryo_fig_dict["colors"][f"tp{x + 1}"] = np.append(embryo_fig_dict["colors"][f"tp{x + 1}"], "#ff1900")
        elif embryo_MCC_predict_dict["match_identity"][f"tp{x + 1}"][i] == "match positive":
            embryo_fig_dict["colors"][f"tp{x + 1}"] = np.append(embryo_fig_dict["colors"][f"tp{x + 1}"], "#0062ff")
        elif embryo_MCC_predict_dict["match_identity"][f"tp{x + 1}"][i] == "match negative":
            embryo_fig_dict["colors"][f"tp{x + 1}"] = np.append(embryo_fig_dict["colors"][f"tp{x + 1}"], "#fafafa")

    #saving the number of false positives, negatives, true positives and negatives
    embryo_MCC_predict_dict["score"][f"tp{x + 1}"] = sklearn.metrics.matthews_corrcoef(embryo_eve_in_dict[f"tp{x + 1}"], embryo_eve_predict_dict["onoff"][f"tp{x + 1}"])
    embryo_MCC_predict_dict["fp"][f"tp{x + 1}"] = pd.DataFrame(np.unique(embryo_MCC_predict_dict["match_identity"][f"tp{x + 1}"], return_counts=True))[1][1]
    embryo_MCC_predict_dict["fn"][f"tp{x + 1}"] = pd.DataFrame(np.unique(embryo_MCC_predict_dict["match_identity"][f"tp{x + 1}"], return_counts=True))[0][1]
    embryo_MCC_predict_dict["tp"][f"tp{x + 1}"] = pd.DataFrame(np.unique(embryo_eve_in_dict[f"tp{x + 1}"], return_counts=True))[1][1]
    embryo_MCC_predict_dict["tn"][f"tp{x + 1}"] = pd.DataFrame(np.unique(embryo_eve_in_dict[f"tp{x + 1}"], return_counts=True))[0][1]

    #making figures
    embryo_fig_dict[f"tp{x+1}"] = go.Figure()
    embryo_fig_dict[f"tp{x + 1}"].add_trace(
        go.Scatter3d(
            x=regression_dataset_dict[f"tp{x+1}"][f"x__{x+1}"],
            y=regression_dataset_dict[f"tp{x+1}"][f"y__{x+1}"],
            z=regression_dataset_dict[f"tp{x+1}"][f"z__{x+1}"],
            mode='markers',
            marker=dict(color=embryo_fig_dict["colors"][f"tp{x + 1}"],
                        size=8
                        )
        )
    )
    embryo_fig_dict[f"tp{x + 1}"].update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                      height=700,
                      width=1200,
                      title = f"Timepoint {x+1}",
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
                                   "<b>True Positive:</b> {TP}<br>"
                                   "<b>False positive:</b> {FP}<br>"
                                   "<b>False negative:</b> {FN}".format(MCC = MCC_predict_dict["score"][f"tp{x+1}"],
                                                                        FP = MCC_predict_dict["fp"][f"tp{x+1}"],
                                                                        FN = MCC_predict_dict["fn"][f"tp{x+1}"],
                                                                        TP = MCC_predict_dict["tp"][f"tp{x+1}"]),
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

#########show figures
#########show figures
embryo_fig_dict["tp1"].show()
embryo_fig_dict["tp2"].show()
embryo_fig_dict["tp3"].show()