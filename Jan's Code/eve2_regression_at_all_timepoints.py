import pandas as pd
import numpy as np


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

#EVE THRESHOLD
eve_threshold = 0.165
#EVE THRESHOLD

position_determinant_dict = {}
for i in range(timepoints):
    position_determinant_dict[f"tp{i+1}"] = [False] * len(regression_dataset_dict[f"tp{i+1}"][f"x__{i+1}"])
    x = 132
    z = 0
    while z <= 160:  # angled section through the embryo because why not; 132-160
        z = 5.7 * x - 5.7 * 132
        x += 0.1
        print(x,z)
        for j in range(len(regression_dataset_dict[f"tp{i+1}"][f"x__{i+1}"])):
            if ((regression_dataset_dict[f"tp{i+1}"][f"x__{i+1}"][j] >= x) &
                (regression_dataset_dict[f"tp{i+1}"][f"x__{i+1}"][j] <= (x+30)) &
                (regression_dataset_dict[f"tp{i+1}"][f"z__{i+1}"][j] >= (z - 0.5)) &
                (regression_dataset_dict[f"tp{i+1}"][f"z__{i+1}"][j] <= (z + 0.5)) &
                (regression_dataset_dict[f"tp{i+1}"][f"eve__{i+1}"][j] >= eve_threshold)):
                position_determinant_dict[f"tp{i + 1}"][j] = True

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
    logreg_models_dict[f"tp{x+1}"] = LogisticRegression(C=1e10)
    logreg_models_dict[f"tp{x + 1}"].fit(os_dict["eve_ex"][f"tp{x+1}"], os_dict["eve_in"][f"tp{x+1}"])
    print(f"timepoint: {x + 1}", logreg_models_dict[f"tp{x + 1}"].coef_)
##coef: ["bcdP__", "bcdP^2",  "hbP__", "gtP__", "KrP__"]
#timepoint: 1[[ 160.92739204 -580.35408109   49.1110909   -44.21172281  -12.00061881]]
#timepoint: 2[[  72.16026578 -328.12243574   28.745335    -45.34698637  -27.5238156 ]]
#timepoint: 3[[  389.88691575 -1232.32363031    43.43848489   -48.99149702  -42.90017515]]

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

for x in range(timepoints):
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
                                   "<b>False negative:</b> {FN}".format(MCC = embryo_MCC_predict_dict["score"][f"tp{x+1}"],
                                                                        FP = embryo_MCC_predict_dict["fp"][f"tp{x+1}"],
                                                                        FN = embryo_MCC_predict_dict["fn"][f"tp{x+1}"],
                                                                        TP = embryo_MCC_predict_dict["tp"][f"tp{x+1}"]),
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
                              borderwidth=1),
                          dict(
                              text="<b><i>Coefficients:</i></b><br>"
                                   "<b>bcdP:</b> {bcdP:.1f}<br>"
                                   "<b>bcdP^2:</b> {bcdP2:.1f}<br>"
                                   "<b>hbP:</b> {hbP:.1f}<br><br>"
                                   "<b>gtP:</b> {gtP:.1f}<br>"
                                   "<b>KrP:</b> {KrP:.1f}<br>".format(bcdP=logreg_models_dict[f"tp{x + 1}"].coef_[0][0],
                                                                      bcdP2=logreg_models_dict[f"tp{x + 1}"].coef_[0][1],
                                                                      hbP=logreg_models_dict[f"tp{x + 1}"].coef_[0][2],
                                                                      gtP=logreg_models_dict[f"tp{x + 1}"].coef_[0][3],
                                                                      KrP=logreg_models_dict[f"tp{x + 1}"].coef_[0][4]),
                              align='left',
                              showarrow=False,
                              xref='paper',
                              yref='paper',
                              x=0.98,
                              y=0.4,
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

########parallel models
########parallel models
########parallel models
########parallel models
########parallel models

parallel = {}
for x in range(timepoints):
    parallel[f"tp{x + 1}"] = {}
    parallel[f"tp{x + 1}"]["eve_ex"] = pd.DataFrame.to_numpy(regression_dataset_dict[f"tp{x + 1}"][[f"bcdP__{x + 1}", f"bcdP^2__{x + 1}", f"hbP__{x + 1}", f"gtP__{x + 1}", f"KrP__{x + 1}"]])
    parallel[f"tp{x + 1}"]["eve_in"] = pd.DataFrame.to_numpy(regression_dataset_dict[f"tp{x + 1}"]["eve_onoff"])

    for y in range(timepoints):
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"] = {}
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_proba"] = logreg_models_dict[f"tp{y + 1}"].predict_proba(parallel[f"tp{x + 1}"]["eve_ex"])

        #determining whether eve is on or of based on the predicted probability
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"] = np.array([])
        for i in range(len(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_proba"])):
            if parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_proba"][i, 1] >= predict_proba_threshold:
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"], True)
            elif parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_proba"][i, 1] < predict_proba_threshold:
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"], False)
            else:
                print("error")
                break

        # determining whether prediction meets original data
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"] = np.array([])
        for i in range(len(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_proba"])):
            # counting the number of false pos, false neg, and true matches
            if parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"][i] != parallel[f"tp{x + 1}"]["eve_in"][i] and parallel[f"tp{x + 1}"]["eve_in"][i] == True:
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"], "false negative")
            elif parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"][i] != parallel[f"tp{x + 1}"]["eve_in"][i] and parallel[f"tp{x + 1}"]["eve_in"][i] == False:
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"], "false positive")
            elif parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"][i] == parallel[f"tp{x + 1}"]["eve_in"][i] and parallel[f"tp{x + 1}"]["eve_in"][i] == True:
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"], "match positive")
            elif parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"][i] == parallel[f"tp{x + 1}"]["eve_in"][i] and parallel[f"tp{x + 1}"]["eve_in"][i] == False:
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"], "match negative")
            else:
                print(error)
                break

        # making color array for plotting below
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"] = np.array([])
        for i in range(len(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_proba"])):
            if parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"][i] == "false positive":
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"], "#f130ff")
            elif parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"][i] == "false negative":
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"], "#ff1900")
            elif parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"][i] == "match positive":
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"], "#0062ff")
            elif parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"][i] == "match negative":
                parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"] = np.append(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"], "#fafafa")

        # saving the number of false positives, negatives, true positives and negatives
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["MCC_score"] = sklearn.metrics.matthews_corrcoef(parallel[f"tp{x + 1}"]["eve_in"], parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["predict_onoff"])
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["fp"] = pd.DataFrame(np.unique(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"], return_counts=True))[1][1]
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["fn"] = pd.DataFrame(np.unique(parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["match_identity"], return_counts=True))[0][1]
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["tp"] = pd.DataFrame(np.unique(parallel[f"tp{x + 1}"]["eve_in"], return_counts=True))[1][1]
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["tn"] = pd.DataFrame(np.unique(parallel[f"tp{x + 1}"]["eve_in"], return_counts=True))[0][1]

        #making figures
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["fig"] = go.Figure()
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["fig"].add_trace(
        go.Scatter3d(
            x=regression_dataset_dict[f"tp{x+1}"][f"x__{x+1}"],
            y=regression_dataset_dict[f"tp{x+1}"][f"y__{x+1}"],
            z=regression_dataset_dict[f"tp{x+1}"][f"z__{x+1}"],
            mode='markers',
            marker=dict(color=parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["colors"],
                        size=8
                        )
        )
        )
        parallel[f"tp{x + 1}"][f"with_model_{y + 1}"]["fig"].update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                       height=700,
                       width=1200,
                       title=f"Timepoint {x + 1}, model:{y+1}",
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
                                    "<b>False negative:</b> {FN}".format(MCC=parallel[f"tp{x+1}"][f"with_model_{y+1}"]["MCC_score"],
                                                                         FP=parallel[f"tp{x+1}"][f"with_model_{y+1}"]["fp"],
                                                                         FN=parallel[f"tp{x+1}"][f"with_model_{y+1}"]["fn"],
                                                                         TP=parallel[f"tp{x+1}"][f"with_model_{y+1}"]["tp"]),
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
                               borderwidth=1),
                           dict(
                               text="<b><i>Coefficients:</i></b><br>"
                                    "<b>bcdP:</b> {bcdP:.1f}<br>"
                                    "<b>bcdP^2:</b> {bcdP2:.1f}<br>"
                                    "<b>hbP:</b> {hbP:.1f}<br><br>"
                                    "<b>gtP:</b> {gtP:.1f}<br>"
                                    "<b>KrP:</b> {KrP:.1f}<br>".format(bcdP = logreg_models_dict[f"tp{y + 1}"].coef_[0][0],
                                                                   bcdP2 = logreg_models_dict[f"tp{y + 1}"].coef_[0][1],
                                                                   hbP = logreg_models_dict[f"tp{y + 1}"].coef_[0][2],
                                                                   gtP = logreg_models_dict[f"tp{y + 1}"].coef_[0][3],
                                                                   KrP = logreg_models_dict[f"tp{y + 1}"].coef_[0][4]),
                               align='left',
                               showarrow=False,
                               xref='paper',
                               yref='paper',
                               x=0.98,
                               y=0.4,
                               bordercolor='black',
                               borderwidth=1)
                           ##ANNOTATIONS
                           ##ANNOTATIONS
                       ]
                       )

parallel[f"tp1"][f"with_model_1"]["fig"].show()
parallel[f"tp1"][f"with_model_2"]["fig"].show()
parallel[f"tp1"][f"with_model_3"]["fig"].show()


parallel[f"tp2"][f"with_model_1"]["fig"].show()
parallel[f"tp2"][f"with_model_2"]["fig"].show()
parallel[f"tp2"][f"with_model_3"]["fig"].show()

parallel[f"tp3"][f"with_model_1"]["fig"].show()
parallel[f"tp3"][f"with_model_2"]["fig"].show()
parallel[f"tp3"][f"with_model_3"]["fig"].show()