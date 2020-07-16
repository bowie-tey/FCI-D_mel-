import pandas as pd
import numpy as np

full_dataset = pd.read_csv("dge/dge_bcdp.csv")
stripes_data = pd.read_csv("dge/Jans_stripes_from_dge_bcdp.csv")
regression_dataset = pd.DataFrame(full_dataset[["X", "Y", "Z", "bcdP", "hb", "Kr", "gt"]])
regression_dataset[["hb", "gt", "Kr"]] = pd.DataFrame(preprocessing.normalize(pd.DataFrame.to_numpy(regression_dataset[["hb", "gt", "Kr"]])))
regression_dataset["bcdP^2"] = np.array(regression_dataset["bcdP"]) ** 2
regression_dataset["eve_onoff"] = stripes_data["stripe_2"]

######################regression
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#balancing
eve_ex = pd.DataFrame.to_numpy(regression_dataset[["bcdP", "bcdP^2", "hb", "gt", "Kr"]])
eve_in = pd.DataFrame.to_numpy(regression_dataset["eve_onoff"])

os = SMOTE(random_state=0)
eve_ex_train, eve_ex_test, eve_in_train, eve_in_test = train_test_split(eve_ex, eve_in, test_size=0.3, random_state=0)

os_data_eve_ex,os_data_eve_in=os.fit_sample(eve_ex_train, eve_in_train)

#model
logreg = LogisticRegression(C=1e10)
logreg.fit(os_data_eve_ex, os_data_eve_in)

#determining the best proba threshold
threshold = []
MCC = []
eve_in_predict_proba = logreg.predict_proba(eve_ex_test)

for j in list(np.arange(0.50, 0.99, 0.01)):
    threshold += [j]
    eve_in_predict = np.array([])
    for i in range(len(eve_ex_test)):
        print(eve_in_predict_proba[i,1])
        if eve_in_predict_proba[i,1] >= j:
            eve_in_predict = np.append(eve_in_predict, True)
        elif eve_in_predict_proba[i,1] < j:
            eve_in_predict = np.append(eve_in_predict, False)
        else:
            print("Error")
            break
    MCC += [sklearn.metrics.matthews_corrcoef(eve_in_test, eve_in_predict)]

MCC_threshold_analysis = pd.DataFrame(list(zip(threshold, MCC)), columns=["threshold", "MCC"])

print("lr score: ", logreg.score(eve_ex_test, eve_in_test), "; ",
          MCC_threshold_analysis.iloc[MCC_threshold_analysis["MCC"].argmax(), :])