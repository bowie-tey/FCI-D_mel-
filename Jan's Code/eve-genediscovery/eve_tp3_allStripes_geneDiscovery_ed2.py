import numpy as np
import pandas as pd

import statistics

import itertools as iter

from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import concurrent.futures

##importing dataset
dataset = pd.read_csv("IMPORT_eve_tp3_allStripes_geneDiscovery_ed2.csv")

##creating all possible combinations of 4 predictors (excluding eve)
gene_combination_columns = list(dataset.columns[:-10])
gene_combination_columns.remove("eve__3")

gene_combinations = list(iter.combinations(gene_combination_columns, 4))

##list of stripes
list_of_stripes = ["1", "2", "5", "3+7", "4+6"]

##median_thresholds; determined externally
median_thresholds = {"1": 0.84,
                     "2": 0.81,
                     "5": 0.80,
                     "3+7": 0.75,
                     "4+6": 0.75}

##number_of_workers
number_of_workers = 16

###modeljudge
###modeljudge
###modeljudge
def model_judge(stripe, gene_combination, dataset, median_thresholds):
    eve_ex = dataset[list(gene_combination)]
    eve_in = dataset[f"stripe_{stripe}"]

    # balancing
    eve_ex_train, eve_ex_test, eve_in_train, eve_in_test = train_test_split(eve_ex, eve_in, test_size=0.3, random_state=0)
    os = SMOTE(random_state=0)
    eve_ex_os, eve_in_os = os.fit_sample(eve_ex_train, eve_in_train)

    # model
    model = LogisticRegression(C=1e10)
    model.fit(eve_ex_os, eve_in_os)

    # predict
    predict_proba = model.predict_proba(eve_ex) #predict proba on the whole dataset

    ##evaluation using the 0.81 threshold
    predict_onoff_081 = np.array([])
    for i in range(len(predict_proba)):
        if predict_proba[i,1] > 0.81:
            predict_onoff_081 = np.append(predict_onoff_081, True)
        elif predict_proba[i,1] <= 0.81:
            predict_onoff_081 = np.append(predict_onoff_081, False)
        else:
            print("error in predict_onoff_081; predict proba if statment failed; lines:55-63")
            break

    ##evaluation using the 0.81 threshold
    predict_onoff_median = np.array([])
    for i in range(len(predict_proba)):
        if predict_proba[i,1] > median_thresholds[stripe]:
            predict_onoff_median = np.append(predict_onoff_median, True)
        elif predict_proba[i,1] <= median_thresholds[stripe]:
            predict_onoff_median = np.append(predict_onoff_median, False)
        else:
            print("error in predict_onoff_median; predict proba if statment failed; lines:67-74")
            break

    #preparing export
    MCC_081 = sklearn.metrics.matthews_corrcoef(eve_in, predict_onoff_081)
    MCC_median = sklearn.metrics.matthews_corrcoef(eve_in, predict_onoff_median)
    coefs = model.coef_
    model = list(gene_combination)
    export = {"Stripe": stripe, "Model": model, "Coefs":coefs, "MCC_081": MCC_081, "MCC_median": MCC_median, "Median_threshold": median_thresholds[stripe]}
    return export
#end of modeljudge

#MULTIPROCESSING
#MULTIPROCESSING
#MULTIPROCESSING
if __name__ == '__main__':
    start = time.time() #start a timer
    print("Start time:", time.ctime(start))

    with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_workers) as executor:
        model_futures = [executor.submit(model_judge, stripe_for_judge, gene_combination_for_judge, dataset, median_thresholds) for stripe_for_judge, gene_combination_for_judge in iter.product(list_of_stripes, gene_combinations[:50])]
        models = pd.DataFrame([future.result() for future in model_futures])

    models.to_csv("EXPORT_eve_tp3_allStripes_geneDiscovery_ed2.csv", index=False)
    print("Finish time:", time.ctime(time.time()))
    print("Start time:", time.ctime(start))
    print(f"Run time: {(time.time() - start)/(60):.2f} min")