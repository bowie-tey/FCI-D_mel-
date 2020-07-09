import pandas as pd
import numpy as np

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

regression_dataset_dict = {}
for x in range(3):
    regression_dataset_dict[f"tp{x+1}"] = pd.DataFrame(dataset[[f"x__{x+1}", f"y__{x+1}", f"z__{x+1}", f"eve__{x+1}", f"bcdP__{x+1}", f"hbP__{x+1}", f"gtP__{x+1}", f"KrP__{x+1}"]])

######################stripe 2 isolation

#THRESHOLD
eve_threshold = 0.165
#THRESHOLD

position_determinant_dict = {}
for x in range(3):
    position_determinant_dict[f"tp{x+1}"] = ((dataset[f"x__{x+1}"] >= 145) & (dataset[f"x__{x+1}"] <= 176) & (dataset[f"z__{x+1}"] >= 70) & (dataset[f"eve__{x+1}"] >= eve_threshold)) | \
                                                                        ((dataset[f"x__{x+1}"] >= 131) & (dataset[f"x__{x+1}"] <= 170) & (dataset[f"z__{x+1}"] < 70) & (dataset[f"eve__{x+1}"] >= eve_threshold))


#adding a "on/off" column to regression_dataset (converts boolean to int)
for x in range(3):
    regression_dataset_dict[list(regression_dataset_dict.keys())[x]]["eve_onoff"] = np.array(list(position_determinant_dict.values())[x])*1


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

for x in range(3):
    eve_ex_dict[f"tp{x + 1}"] = pd.DataFrame.to_numpy(regression_dataset_dict[list(regression_dataset_dict.keys())[x]][[f"bcdP__{x+1}", f"hbP__{x+1}", f"gtP__{x+1}", f"KrP__{x+1}"]])
    eve_in_dict[f"tp{x + 1}"] = pd.DataFrame.to_numpy(regression_dataset_dict[list(regression_dataset_dict.keys())[x]]["eve_onoff"])

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

for x in range(3):
    logreg_models_dict[f"tp{x+1}"] = LogisticRegression()
    logreg_models_dict[f"tp{x + 1}"].fit(os_dict["eve_ex"][f"tp{x+1}"], os_dict["eve_in"][f"tp{x+1}"])
    print(f"timepoint: {x + 1}", logreg_models_dict[f"tp{x + 1}"].coef_)

##coef: ["bcdP__", "hbP__", "gtP__", "KrP__"]
#timepoint: 1[[-19.36660166  17.26118287 - 13.03516797 - 3.90388066]]
#timepoint: 2[[-16.46772599  13.86282708 - 11.20048436 - 4.28701761]]
#timepoint: 3[[-20.30643861  16.43566713 - 10.54650913 - 2.99629309]]