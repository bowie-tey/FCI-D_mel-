import numpy as np
import pandas as pd
import matplotlib

import plotly
import plotly.graph_objects as go

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

##subseting for timepoit 3
timepoint6_dataset = pd.DataFrame(dataset.iloc[:,dataset.columns.get_loc("x__6"):])

##########threshold determination
##########threshold determination
##########threshold determination

#timepoint5_dataset["ftz__5"].plot.hist(bins=40)

#FTZ THRESHOLD
ftz_threshold = 0.165 #based on visual inspection down below
#FTZ THRESHOLD

active_cells = []
for i in list(timepoint6_dataset["ftz__6"]):
    if i > ftz_threshold:
        active_cells += [1]
    else:
        active_cells += [0]

threshold_fig = go.Figure()
threshold_fig.add_trace(
        go.Scatter3d(
            x=timepoint6_dataset["x__6"],
            y=timepoint6_dataset["y__6"],
            z=timepoint6_dataset["z__6"],
            mode='markers',
            marker=dict(color=active_cells,
                        size=8,
                        colorscale="blues",
                        )
        )
    )
threshold_fig.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                            title_text="Ftz Stripes",
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
#stripe_fig.show()

##########stripe selection
##########stripe selection
##########stripe selection

stripes = {}
for i in (np.array(range(7))+1): #numbers of stripes
    stripes[str(i)] = [0] * len(timepoint6_dataset["ftz__6"])

start_x = {"1": 108, "2": 145, "3": 185, "4": 226, "5": 260, "6": 305, "7": 344} #used to determine dela_X and stripe position
end_x = {"1": 125, "2": 162, "3": 195, "4": 220, "5": 247, "6": 275, "7": 312} #used to determine dela_X
plus_x = {"1": 140-108, "2": 179-145, "3": 215-185, "4": 257-226, "5": 295-260, "6": 342-305, "7": 375-344} #stripe thikness

slope = {} #steepness of selection
for i in (np.array(range(7))+1): #numbers of stripes
    slope[str(i)] = 160/(end_x[str(i)]-start_x[str(i)])


z = {}
x = {"1": 108, "2": 145, "3": 185, "4": 226, "5": 260, "6": 305, "7": 344} # identical to start_x - will change through out the loop
for i in (np.array(range(7))+1):
    z[str(i)] = 0

while min(abs(np.array(list(z.values())))) < 160:
    print(min(abs(np.array(list(z.values())))))
    for i in stripes.keys():
        z[str(i)] = slope[str(i)] * x[str(i)] - slope[str(i)] * start_x[str(i)]
        if slope[str(i)] >0:
            x[str(i)] += 0.1
        elif slope[str(i)] < 0:
            x[str(i)] -= 0.1

        for j in range(len(timepoint6_dataset["ftz__6"])):
            if ((timepoint6_dataset["x__6"][j] > (x[str(i)])) & (timepoint6_dataset["x__6"][j] < (x[str(i)] + plus_x[str(i)])) & \
                (timepoint6_dataset["z__6"][j] > (z[str(i)]-2)) & (timepoint6_dataset["z__6"][j] < (z[str(i)] + 2)) & (timepoint6_dataset["ftz__6"][j] > ftz_threshold)):
                stripes[str(i)][j] = 1
            else:
                pass

#making a single dolumn specifing cell stripe identity
all_stripes = [0] * len(timepoint6_dataset["ftz__6"])
for i in stripes.keys():
    for j in range(len(timepoint6_dataset["ftz__6"])):
        if stripes[i][j] == 1 and all_stripes[j] == 0:
            all_stripes[j] = int(i)
        elif stripes[i][j] == 0:
            pass
        else:
            print("error", i, all_stripes[j])
            all_stripes[j] = int(i)


#exporting to csv
columns = list(zip(stripes["1"], stripes["2"], stripes["3"], stripes["4"], stripes["5"], stripes["6"], stripes["7"], all_stripes))
column_names = ["stripe_1", "stripe_2", "stripe_3", "stripe_4", "stripe_5", "stripe_6", "stripe_7", "all_stripes"]
stripes_export = pd.DataFrame(columns, columns = column_names)
stripes_export.to_csv("ftz\Jans_ftz_stripes_from_bdtnp_tp6.csv", index=False)

####PLOTING TO VALIDATE
####PLOTING TO VALIDATE
####PLOTING TO VALIDATE
####PLOTING TO VALIDATE

#making color vector based on stripe identity
all_stripes = pd.read_csv("ftz/Jans_ftz_stripes_from_bdtnp.csv")["all_stripes"]
colors = ["#fafafa"]*len(all_stripes) #sets default color to white
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
        colors[i] = "#f6ff54"
    elif all_stripes[i] == 6:
        colors[i] = "#6d32a8"
    elif all_stripes[i] == 7:
        colors[i] = "#00ff91"
    elif all_stripes[i] != 0:
        print("error")
        break

validation_figure = go.Figure()
validation_figure.add_trace(
    go.Scatter3d(
        x=timepoint6_dataset["x__6"],
        y=timepoint6_dataset["y__6"],
        z=timepoint6_dataset["z__6"],
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
