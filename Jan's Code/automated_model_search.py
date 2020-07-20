import numpy as np
import pandas as pd

import itertools as int

from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



###TIMEPOINTS TO ANALYSE
timepoints = 3
###TIMEPOINTS TO ANALYSE

###nO of stripes
stripes = 7
###nO of stripes

##PREDICT PROBA TRESHOLD
predict_proba_threshold = 0.81
##PREDICT PROBA TRESHOLD

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")
stripe_data = pd.read_csv("Jans_stripes_from_bdtnp.csv")

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

##subseting for timepoit 3
timepoint3_dataset = pd.DataFrame(dataset.iloc[:,dataset.columns.get_loc("x__3"):dataset.columns.get_loc("x__4")])
timepoint3_dataset["stripe"] = stripe_data["all_stripes"]

###############COMBINATIONS
###############COMBINATIONS

#first 10 columns do not include exoression information
timepoint3_dataset_gene_columns = list(timepoint3_dataset.columns[10:])

#removing eve
timepoint3_dataset_gene_columns_noeve = timepoint3_dataset_gene_columns
timepoint3_dataset_gene_columns_noeve.remove("eve__3")

##creating all possible combinations
gene_combinations = list(int.combinations(timepoint3_dataset_gene_columns_noeve, 3)) + \
                    list(int.combinations(timepoint3_dataset_gene_columns_noeve, 4))

###############MODELLING
###############MODELLING
###############MODELLING

def model_judge(i, x):
    eve_ex = timepoint3_dataset[list(i)]
    eve_in = timepoint3_dataset[f"eve_st{x + 1}"]

    # balancing
    eve_ex_train, eve_ex_test, eve_in_train, eve_in_test = train_test_split(eve_ex, eve_in, test_size=0.3, random_state=0)

    os = SMOTE(random_state=0)

    eve_ex_os, eve_in_os = os.fit_sample(eve_ex_train, eve_in_train)

    # model
    model = LogisticRegression(C=1e10)
    model.fit(eve_ex_os, eve_in_os)

    # predict
    predict_proba = model.predict_proba(eve_ex_test)
    predict_onoff = np.array([])

    #decide whether the cell is eve on/off based on the prediction
    # determining the best proba threshold
    threshold = []
    MCC = []

    for j in list(np.arange(0.50, 0.95, 0.01)):
        threshold += [j]
        predict_onoff = np.array([])
        for e in range(len(predict_proba)):
            if predict_proba[e, 1] >= j:
                predict_onoff = np.append(predict_onoff, True)
            elif predict_proba[e, 1] < j:
                predict_onoff = np.append(predict_onoff, False)
            else:
                print("error: predict proba -> onoff")
                break
                break
                break
                break
        # test
        MCC += [sklearn.metrics.matthews_corrcoef(eve_in_test, predict_onoff)]

    # export
    MCC_threshold_analysis = pd.DataFrame(list(zip(threshold, MCC)), columns=["threshold", "MCC"])
    export = MCC_threshold_analysis.iloc[MCC_threshold_analysis["MCC"].argmax()]
    return {"model": i, "stripe": x+1, str(export.index[0]): export[0], str(export.index[1]): export[1]}

###############MODELLING
###############MODELLING
###############MODELLING

####multiprocessing
from time import time
import concurrent.futures

if __name__ == '__main__':
    start = time() #start a timer
    print(start)

    with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
        model_futures = [executor.submit(model_judge, i_judge, x_judge) for i_judge, x_judge in int.product(gene_combinations, [0,2,3,4,5,6])]
        models = pd.DataFrame([future.result() for future in model_futures])

    models.to_csv("automated_model_search.csv", index=False)
    print(f"run time: {time() - start}")