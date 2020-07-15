#READ ME#
#reads the "dge_dcdp.csv" and subsets individuall cells into 7 stripes based on their position and eve expression
#results are saved to "Jans_stripes_from_dge_bcdp.csv"
#plotly is used for visual inspection
import pandas as pd
import numpy as np

import plotly
import plotly.graph_objects as go

full_data = pd.read_csv("dge_bcdp.csv")
dataset = pd.DataFrame(full_data[["eve", "X", "Y", "Z"]])

#normalising coordinates to 0
dataset["X"] -= min(dataset["X"])
dataset["Y"] -= min(dataset["Y"])
dataset["Z"] -= min(dataset["Z"])

#thresholding
eve_threshold = 0.000499
eve_onoff = [0] * len(dataset["eve"])

for i in range(len(dataset["eve"])):
    if dataset["eve"][i] >= eve_threshold:
        eve_onoff[i] = 1
    else:
        pass
dataset["eve_onoff"] = eve_onoff

initial_figure = go.Figure()
initial_figure.add_trace(
    go.Scatter3d(
        x = dataset["X"],
        y = dataset["Y"],
        z = dataset["Z"],
        mode='markers',
        marker=dict(color=dataset["eve_onoff"],
                        colorscale="blues",
                        size=8
                        )
    )
)
initial_figure.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
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
initial_figure.show()

#########SUBSETTING STRIPES
#########SUBSETTING STRIPES
#########SUBSETTING STRIPES
#########SUBSETTING STRIPES
#########SUBSETTING STRIPES

stripes = {}
for i in (np.array(range(7))+1):
    stripes[str(i)] = [0] * len(dataset["eve"])

start_x = {"1": 91, "2": 123, "3": 161, "4": 199, "5": 243, "6": 281, "7": 326} #used to determine dela_X and stripe position
end_x = {"1": 114, "2": 147, "3": 183, "4": 206, "5": 235, "6": 259, "7": 289} #used to determine dela_X
plus_x = {"1": 128-91, "2": 173-147, "3": 187-161, "4": 225-199, "5": 269-243, "6": 308-281, "7": 350-326} #stripe thikness

slope = {} #steepness of selection
for i in (np.array(range(7))+1):
    slope[str(i)] = 160/(end_x[str(i)]-start_x[str(i)])


z = {}
x = {"1": 91, "2": 123, "3": 161, "4": 199, "5": 243, "6": 281, "7": 326}
for i in (np.array(range(7))+1):
    z[str(i)] = 0

while min(abs(np.array(list(z.values())))) < 160:
    for i in stripes.keys():
        z[str(i)] = slope[str(i)] * x[str(i)] - slope[str(i)] * start_x[str(i)]
        if slope[str(i)] >0:
            x[str(i)] += 0.1
        elif slope[str(i)] < 0:
            x[str(i)] -= 0.1

        for j in range(len(dataset["eve"])):
            if ((dataset["X"][j] > (x[str(i)])) & (dataset["X"][j] < (x[str(i)] + plus_x[str(i)])) & \
                (dataset["Z"][j] > (z[str(i)]-2)) & (dataset["Z"][j] < (z[str(i)] + 2)) & (dataset["eve_onoff"][j] == True)):
                stripes[str(i)][j] = 1
            else:
                pass

#making a single dolumn specifing cell stripe identity
all_stripes = [0] * len(dataset["eve"])
for i in stripes.keys():
    for j in range(len(dataset["eve"])):
        if stripes[i][j] == 1 and all_stripes[j] == 0:
            all_stripes[j] = int(i)
        elif stripes[i][j] == 0:
            pass
        else:
            print("error", i, all_stripes[j])
            break


#exporting to csv
columns = list(zip(stripes["1"], stripes["2"], stripes["3"], stripes["4"], stripes["5"], stripes["6"], stripes["7"], all_stripes))
column_names = ["stripe_1", "stripe_2", "stripe_3", "stripe_4", "stripe_5", "stripe_6", "stripe_7", "all_stripes"]
stripes_export = pd.DataFrame(columns, columns = column_names)
stripes_export.to_csv("Jans_stripes_from_dge_bcdp.csv", index=False)

####PLOTING TO VALIDATE
####PLOTING TO VALIDATE
####PLOTING TO VALIDATE
####PLOTING TO VALIDATE

#making color vector based on stripe identity
colors = ["#fafafa"]*len(dataset["eve"]) #sets default color to white
for i in range(len(all_stripes)):
    if all_stripes[i] == 1:
        colors[i] = "#ffaa00"
    elif all_stripes[i] == 2:
        colors[i] = "#ff2a00"
    elif all_stripes[i] == 3:
        colors[i] = "#0091ff"
    elif all_stripes[i] == 4:
        colors[i] = "#ff00ea"
    elif all_stripes[i] == 5:
        colors[i] = "#1eff00"
    elif all_stripes[i] == 6:
        colors[i] = "#ff005d"
    elif all_stripes[i] == 7:
        colors[i] = "#00ff91"
    elif all_stripes[i] != 0:
        print("error")
        break

validation_figure = go.Figure()
validation_figure.add_trace(
    go.Scatter3d(
        x=dataset["X"],
        y=dataset["Y"],
        z=dataset["Z"],
        mode='markers',
        marker=dict(color=colors,
                    size=8
                    )
    )
)
validation_figure.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
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
validation_figure.show()
print("finished")