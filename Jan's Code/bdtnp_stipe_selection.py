import numpy as np
import pandas as pd

import plotly
import plotly.graph_objects as go

###nO of stripes
stripes = 7
###nO of stripes

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

##subseting for timepoit 3
timepoint3_dataset = pd.DataFrame(dataset.iloc[:,dataset.columns.get_loc("x__3"):dataset.columns.get_loc("x__4")])

##########stripe selection
##########stripe selection
##########stripe selection


#EVE THRESHOLD
eve_threshold = 0.165
#EVE THRESHOLD

onoff_determinant_dict = {"st1": {}, "st2": {}, "st3": {}, "st4": {}, "st5": {}, "st6": {}, "st7": {}}
#stripe1: x 100,145 & 90,131; z 70
onoff_determinant_dict["st1"]["bol"] = (((timepoint3_dataset["x__3"] >= 100) & \
                                         (timepoint3_dataset["x__3"] < 145) & \
                                         (timepoint3_dataset["z__3"] >= 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)) | \
                                        ((timepoint3_dataset["x__3"] >= 90) & \
                                         (timepoint3_dataset["x__3"] < 131) & \
                                         (timepoint3_dataset["z__3"] < 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)))
onoff_determinant_dict["st1"]["num"] = np.array(onoff_determinant_dict["st1"]["bol"])*1

#stripe2: x 145,185 & 131,175; z 70
onoff_determinant_dict["st2"]["bol"] = (((timepoint3_dataset["x__3"] >= 145) & \
                                         (timepoint3_dataset["x__3"] < 185) & \
                                         (timepoint3_dataset["z__3"] >= 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)) | \
                                        ((timepoint3_dataset["x__3"] >= 131) & \
                                         (timepoint3_dataset["x__3"] < 175) & \
                                         (timepoint3_dataset["z__3"] < 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)))
onoff_determinant_dict["st2"]["num"] = np.array(onoff_determinant_dict["st2"]["bol"])*1

#stripe3: x 185,215 & 175,210; z 70
onoff_determinant_dict["st3"]["bol"] = (((timepoint3_dataset["x__3"] >= 185) & \
                                         (timepoint3_dataset["x__3"] < 215) & \
                                         (timepoint3_dataset["z__3"] >= 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)) | \
                                        ((timepoint3_dataset["x__3"] >= 175) & \
                                         (timepoint3_dataset["x__3"] < 210) & \
                                         (timepoint3_dataset["z__3"] < 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)))
onoff_determinant_dict["st3"]["num"] = np.array(onoff_determinant_dict["st3"]["bol"])*1

#stripe4 x 215,245 & 210,250; z 70
onoff_determinant_dict["st4"]["bol"] = (((timepoint3_dataset["x__3"] >= 215) & \
                                         (timepoint3_dataset["x__3"] < 245) & \
                                         (timepoint3_dataset["z__3"] >= 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)) | \
                                        ((timepoint3_dataset["x__3"] >= 210) & \
                                         (timepoint3_dataset["x__3"] < 250) & \
                                         (timepoint3_dataset["z__3"] < 70) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)))
onoff_determinant_dict["st4"]["num"] = np.array(onoff_determinant_dict["st4"]["bol"])*1

#stripe5 x 245,274 & 250,286; z 60
onoff_determinant_dict["st5"]["bol"] = (((timepoint3_dataset["x__3"] >= 245) & \
                                         (timepoint3_dataset["x__3"] < 274) & \
                                         (timepoint3_dataset["z__3"] >= 60) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)) | \
                                        ((timepoint3_dataset["x__3"] >= 250) & \
                                         (timepoint3_dataset["x__3"] < 286) & \
                                         (timepoint3_dataset["z__3"] < 60) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)))
onoff_determinant_dict["st5"]["num"] = np.array(onoff_determinant_dict["st5"]["bol"])*1

#stripe6 x 274,312 & 286,326; z 60
onoff_determinant_dict["st6"]["bol"] = (((timepoint3_dataset["x__3"] >= 274) & \
                                         (timepoint3_dataset["x__3"] < 312) & \
                                         (timepoint3_dataset["z__3"] >= 60) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)) | \
                                        ((timepoint3_dataset["x__3"] >= 286) & \
                                         (timepoint3_dataset["x__3"] < 326) & \
                                         (timepoint3_dataset["z__3"] < 60) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)))
onoff_determinant_dict["st6"]["num"] = np.array(onoff_determinant_dict["st6"]["bol"])*1

#stripe7 x 312,360 & 326,380; z 60
onoff_determinant_dict["st7"]["bol"] = (((timepoint3_dataset["x__3"] >= 312) & \
                                         (timepoint3_dataset["x__3"] < 360) & \
                                         (timepoint3_dataset["z__3"] >= 60) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)) | \
                                        ((timepoint3_dataset["x__3"] >= 326) & \
                                         (timepoint3_dataset["x__3"] < 380) & \
                                         (timepoint3_dataset["z__3"] < 60) & \
                                         (timepoint3_dataset["eve__3"] >= eve_threshold)))
onoff_determinant_dict["st7"]["num"] = np.array(onoff_determinant_dict["st7"]["bol"])*1

#adding columns for each stripe to the timepoint3_dataset
for x in range(stripes):
    timepoint3_dataset[f"eve_st{x+1}"] = onoff_determinant_dict[f"st{x+1}"]["num"]

#making a single dolumn specifing cell stripe identity
stripe = [0]*len(timepoint3_dataset["x__3"])

for x in range(stripes):
    for i in range(len(onoff_determinant_dict[f"st{x+1}"]["num"])):
        if len(stripe) != len(onoff_determinant_dict[f"st{x+1}"]["num"]):
            print("error: lengths don't match")
            break

        if onoff_determinant_dict[f"st{x+1}"]["num"][i] == 1 and stripe[i] == 0:
            stripe[i] = x+1
        elif onoff_determinant_dict[f"st{x+1}"]["num"][i] == 0 and stripe[i] == 0:
            stripe[i] = 0
        elif onoff_determinant_dict[f"st{x+1}"]["num"][i] == 0:
            pass
        elif stripe[i] != 0:
            print("error: stripe cell overlap")
            print(x+1, stripe[i], "x:", timepoint3_dataset["x__3"][i], "y:", timepoint3_dataset["y__3"][i], "z:", timepoint3_dataset["z__3"][i])
        else:
            print("error: other")
            break

#exporting stripe identities
columns = list(zip(onoff_determinant_dict["st1"]["num"], onoff_determinant_dict["st2"]["num"], onoff_determinant_dict["st3"]["num"],
                   onoff_determinant_dict["st4"]["num"], onoff_determinant_dict["st5"]["num"], onoff_determinant_dict["st6"]["num"],
                   onoff_determinant_dict["st7"]["num"], stripe))
column_names = ["stripe_1", "stripe_2", "stripe_3", "stripe_4", "stripe_5", "stripe_6", "stripe_7", "all_stripes"]
stripes_export = pd.DataFrame(columns, columns = column_names)
stripes_export.to_csv("Jans_stripes_from_bdtnp.csv", index=False)

####ploting stripe selection
####ploting stripe selection

#making color vector based on stripe identity
timepoint3_dataset["stripe"] = stripe
colors = ["#fafafa"]*len(timepoint3_dataset["x__3"]) #sets default color to white

for i in range(len(timepoint3_dataset["stripe"])):
    if timepoint3_dataset["stripe"][i] == 1:
        colors[i] = "#ffaa00"
    elif timepoint3_dataset["stripe"][i] == 2:
        colors[i] = "#ff2a00"
    elif timepoint3_dataset["stripe"][i] == 3:
        colors[i] = "#0091ff"
    elif timepoint3_dataset["stripe"][i] == 4:
        colors[i] = "#ff00ea"
    elif timepoint3_dataset["stripe"][i] == 5:
        colors[i] = "#1eff00"
    elif timepoint3_dataset["stripe"][i] == 6:
        colors[i] = "#ff005d"
    elif timepoint3_dataset["stripe"][i] == 7:
        colors[i] = "#00ff91"
    elif timepoint3_dataset["stripe"][i] != 0:
        print("error")
        break

stripe_fig = go.Figure()
stripe_fig.add_trace(
        go.Scatter3d(
            x=timepoint3_dataset["x__3"],
            y=timepoint3_dataset["y__3"],
            z=timepoint3_dataset["z__3"],
            mode='markers',
            marker=dict(color=colors,
                        size=8
                        )
        )
    )
stripe_fig.update_layout(scene_camera=dict(eye=dict(x=0, y=-1.8, z=0.6)),
                  title_text="Eve Stripes",
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