import pandas as pd
import numpy as np

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

regression_dataset = pd.DataFrame(dataset[["id", "x__3", "y__3", "z__3","eve__3", "bcdP__3", "hbP__3", "gtP__3", "KrP__3"]])

######################stripe 2 isolation
eve_threshold = 0.165
position_determinant = ((dataset["x__3"] >= 145) & (dataset["x__3"] <= 176) & (dataset["z__3"] >= 70) & (dataset["eve__3"] >= eve_threshold)) | \
                       ((dataset["x__3"] >= 131) & (dataset["x__3"] <= 170) & (dataset["z__3"] < 70) & (dataset["eve__3"] >= eve_threshold))
#stripe2_cells = dataset[position_determinant]

#adding a "on/off" column to regression_dataset
regression_dataset["eve_onoff"] = np.array(position_determinant)*1


######################regression
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#balancing
eve_ex = pd.DataFrame.to_numpy(regression_dataset[["bcdP__3", "hbP__3", "gtP__3", "KrP__3"]])
eve_in = pd.DataFrame.to_numpy(regression_dataset["eve_onoff"])

os = SMOTE(random_state=0)
eve_ex_train, eve_ex_test, eve_in_train, eve_in_test = train_test_split(eve_ex, eve_in, test_size=0.3, random_state=0)

os_data_eve_ex,os_data_eve_in=os.fit_sample(eve_ex_train, eve_in_train)
os_data_eve_ex = pd.DataFrame(data=os_data_eve_ex,columns=["bcdP__3", "hbP__3", "gtP__3", "KrP__3"])
os_data_eve_in= pd.DataFrame(data=os_data_eve_in,columns=["eve_onoff"])

#model
logreg = LogisticRegression()
logreg.fit(pd.DataFrame.to_numpy(os_data_eve_ex), pd.DataFrame.to_numpy(os_data_eve_in).ravel())
#coef - [[-20.30643861  16.43566713 -10.54650913  -2.99629309]]; ["bcdP__3", "hbP__3", "gtP__3", "KrP__3"]

#verifying the validity of the model /w MCC
eve_in_predict_proba = logreg.predict_proba(eve_ex_test)
eve_in_predict = np.array([])
for i in range(1824):
    if eve_in_predict_proba[i,1] >= 0.81:
        eve_in_predict = np.append(eve_in_predict, True)
    elif eve_in_predict_proba[i,1] < 0.81:
        eve_in_predict = np.append(eve_in_predict, False)
    else:
        print("Error")
        break
sklearn.metrics.matthews_corrcoef(eve_in_test, eve_in_predict)

#counting the number of false pos, false neg, and true matches
miss = np.array([])
for i in range(1824):
    if eve_in_predict[i] != eve_in_test[i] and eve_in_test[i] == True:
        miss = np.append(miss, "Neg")
    elif eve_in_predict[i] != eve_in_test[i] and eve_in_test[i] == False:
        miss = np.append(miss, "Pos")
    else:
        miss = np.append(miss, "a")
np.asarray(np.unique(miss, return_counts=True)).T

######################headless
headless_regression_dataset = pd.DataFrame(regression_dataset)

#removing head cells
x = 90
z = 0
while z <= 160: #angled section through the embryo because why not
    z = 8*x - 8*90
    x += 0.05
    print(x, z)
    headless_regression_dataset = headless_regression_dataset.drop(headless_regression_dataset[((headless_regression_dataset["x__3"] <= x) &
                                                                  ((headless_regression_dataset["z__3"] >= (z-0.1)) & (headless_regression_dataset["z__3"] <= (z+0.1))))].index)

##stripe 2 isolation
headless_position_determinant = ((headless_regression_dataset["x__3"] >= 145) & (headless_regression_dataset["x__3"] <= 176) &
                                 (headless_regression_dataset["z__3"] >= 70) & (headless_regression_dataset["eve__3"] >= eve_threshold)) | \
                                ((headless_regression_dataset["x__3"] >= 131) & (headless_regression_dataset["x__3"] <= 170) &
                                 (headless_regression_dataset["z__3"] < 70) & (headless_regression_dataset["eve__3"] >= eve_threshold))
headless_regression_dataset["eve_onoff"] = np.array(headless_position_determinant)*1

###regression
#balancing
headless_eve_ex = pd.DataFrame.to_numpy(headless_regression_dataset[["bcdP__3", "hbP__3", "gtP__3", "KrP__3"]])
headless_eve_in = pd.DataFrame.to_numpy(headless_regression_dataset["eve_onoff"])

os = SMOTE(random_state=0)
headless_eve_ex_train, headless_eve_ex_test, headless_eve_in_train, headless_eve_in_test = train_test_split(headless_eve_ex, headless_eve_in, test_size=0.3, random_state=0)

os_data_headless_eve_ex,os_data_headless_eve_in=os.fit_sample(headless_eve_ex_train, headless_eve_in_train)
os_data_headless_eve_ex = pd.DataFrame(data=os_data_headless_eve_ex,columns=["bcdP__3", "hbP__3", "gtP__3", "KrP__3"])
os_data_headless_eve_in= pd.DataFrame(data=os_data_headless_eve_in,columns=["eve_onoff"])

#model
headless_logreg = LogisticRegression()
headless_logreg.fit(pd.DataFrame.to_numpy(os_data_headless_eve_ex), pd.DataFrame.to_numpy(os_data_headless_eve_in).ravel())
#coef: [[  0.25723279  16.00146691 -13.87495209  -6.16236073]]; ["bcdP__3", "hbP__3", "gtP__3", "KrP__3"]

######################ploting
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt

fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=headless_regression_dataset["x__3"],
        y=headless_regression_dataset["y__3"],
        z=headless_regression_dataset["z__3"],
        name="1",
        mode='markers',
        visible=True,
        marker=dict(color=headless_regression_dataset["eve__3"],
                    colorscale="blues",
                    cmin=0,
                    cmax=1,
                    size=8
                    )
    )
)

#fig.add_trace(
#    go.Scatter3d(
#        x=regression_dataset["x__3"][regression_dataset["eve__3"] >= eve_threshold],
#        y=regression_dataset["y__3"][regression_dataset["eve__3"] >= eve_threshold],
#        z=regression_dataset["z__3"][regression_dataset["eve__3"] >= eve_threshold],
#        name="1",
#        mode='markers',
#        visible=True,
#        marker=dict(color=regression_dataset["eve__3"][regression_dataset["eve__3"] >= eve_threshold],
#                    colorscale="purples",
#                    cmin=0,
#                    cmax=1,
#                    size=8,
#                    opacity=1
#                    )
#    )
#)
fig.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
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
fig.show()