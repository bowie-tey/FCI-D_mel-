import numpy as np
import pandas as pd

import statistics

import itertools

from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

######################dataset
dataset = pd.read_csv("D_mel_atlas.csv")
stripe_data = pd.read_csv("ftz/Jans_ftz_stripes_from_bdtnp_tp6.csv")

#creating combo stripes - 1,2,5 & 3,6,7
combo_stripe12 = []
combo_stripe467 = []
for i in list(stripe_data["all_stripes"]):
    if (i == 4) or (i == 6) or (i == 7):
        combo_stripe467 += [1]
        combo_stripe12 += [0]
    elif (i == 1) or (i == 2):
        combo_stripe467 += [0]
        combo_stripe12 += [1]
    else:
        combo_stripe467 += [0]
        combo_stripe12 += [0]
stripe_data["stripe_1+2"] = combo_stripe12
stripe_data["stripe_4+6+7"] = combo_stripe467

#normalising coordinates to 0
for x in range(6):
    dataset[f"x__{x+1}"] -= min(dataset[f"x__{x+1}"])
    dataset[f"y__{x+1}"] -= min(dataset[f"y__{x+1}"])
    dataset[f"z__{x+1}"] -= min(dataset[f"z__{x+1}"])

##subseting for timepoit 6
timepoint6_dataset = pd.DataFrame(dataset.iloc[:, dataset.columns.get_loc("x__6"):-9])

#first 6 columns do not include exoression information
timepoint6_dataset_gene_columns = list(timepoint6_dataset.columns[6:])

#removing columns that are not needed/are redundant
timepoint6_dataset_gene_columns_refined = timepoint6_dataset_gene_columns
timepoint6_dataset_gene_columns_refined.remove("Kr__6")
timepoint6_dataset_gene_columns_refined.remove("gt__6")
timepoint6_dataset_gene_columns_refined.remove("hb__6")
timepoint6_dataset_gene_columns_refined.remove("ftz__6")

#adding stripe columns
timepoint6_dataset = pd.concat([timepoint6_dataset, stripe_data], axis=1, sort=False)

##creating all possible combinations
gene_combinations = list(itertools.combinations(timepoint6_dataset_gene_columns_refined, 4))

###############MODELLING
###############MODELLING
###############MODELLING

def model_judge(i, x):
    models_export_list = []
    median_data = {}

    ftz_ex = timepoint6_dataset[list(i)]
    ftz_in = timepoint6_dataset[f"stripe_{x}"]

    # balancing
    ftz_ex_train, ftz_ex_test, ftz_in_train, ftz_in_test = train_test_split(ftz_ex, ftz_in, test_size=0.3, random_state=0)

    os = SMOTE(random_state=0)

    ftz_ex_os, ftz_in_os = os.fit_sample(ftz_ex_train, ftz_in_train)

    # model
    model = LogisticRegression(C=1e10)
    model.fit(ftz_ex_os, ftz_in_os)

    # predict
    predict_proba = model.predict_proba(ftz_ex)

    #decide whether the cell is eve on/off based on the prediction
    # determining the best proba threshold
    threshold = []
    MCC = []

    for j in list(np.round(np.arange(0.70, 0.95, 0.01), 2)):
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
        # test
        MCC += [sklearn.metrics.matthews_corrcoef(ftz_in, predict_onoff)]


    MCC_threshold_analysis = pd.DataFrame(list(zip(threshold, MCC)), columns=["threshold", "MCC"])
    export = MCC_threshold_analysis.iloc[MCC_threshold_analysis["MCC"].argmax()]

    # export
    median_data[f"{i}_{x}"] = {"predict_proba": predict_proba, "ftz_in": ftz_in}
    models_export_list += [{"model": i, "stripe": x, f"{export.index[0]}": export[0], f"{export.index[1]}": export[1], "Coefs": model.coef_}]

    models_export_dataframe = pd.DataFrame(models_export_list)
    return_data = {"median_data": median_data, "dataframe": models_export_dataframe}
    return return_data

def merge_dict(dict1, dict2): #merges two dictoneries into one
    return {**dict1, **dict2}

def MCC_judge(i, x, median_proba_threshold, median_data_to_judge):
    proba_threshold = median_proba_threshold[x]
    predict_proba = median_data_to_judge[f"{i}_{x}"]["predict_proba"]
    ftz_in = median_data_to_judge[f"{i}_{x}"]["ftz_in"]

    predict_onoff = np.array([])
    for e in range(len(predict_proba)):
        if predict_proba[e, 1] >= proba_threshold:
            predict_onoff = np.append(predict_onoff, True)
        elif predict_proba[e, 1] < proba_threshold:
            predict_onoff = np.append(predict_onoff, False)
        else:
            print("error: predict proba -> onoff")
            break

    return {"model": i, "stripe": x, "median_threshold": proba_threshold, "median_MCC": sklearn.metrics.matthews_corrcoef(ftz_in, predict_onoff)}

####multiprocessing
####multiprocessing
####multiprocessing
import time
import concurrent.futures

stripe_list = ["3", "5", "1+2", "4+6+7"]

if __name__ == '__main__':
    start = time.time() #start a timer
    print("Start time:", time.ctime(start))

    with concurrent.futures.ProcessPoolExecutor(max_workers=11) as executor:
        model_futures = [executor.submit(model_judge, i_judge, x_judge) for i_judge, x_judge in itertools.product(gene_combinations[:1000], stripe_list)]
        model_future_results = [future.result() for future in model_futures]
    print("part1 done")
    print(f"Run time(1): {(time.time() - start) / (60):.2f} minutes")

    #extract the dataframes from model judge
    dataframes_from_futures = []
    for y in model_future_results:
        dataframes_from_futures += [y["dataframe"]]
    dataframes_from_futures = pd.concat(dataframes_from_futures, ignore_index=True, sort=False)

    #extract "median_data" from model judge
    median_data_to_judge = {}
    for y in model_future_results:
        median_data_to_judge = merge_dict(median_data_to_judge, y["median_data"])

    #determining the best threshold
    median_proba_threshold = {}
    for x in stripe_list:
        median_proba_threshold_determinant = []
        median_proba_threshold_threshold = list(dataframes_from_futures["threshold"][dataframes_from_futures["stripe"] == x])
        median_proba_threshold_MCC = list(dataframes_from_futures["MCC"][dataframes_from_futures["stripe"] == x])
        for y in range(len(median_proba_threshold_threshold)):
            if median_proba_threshold_MCC[y] > 0.2: #subselect for stripes
                median_proba_threshold_determinant += [median_proba_threshold_threshold[y]]
        median_proba_threshold[x] = statistics.median(median_proba_threshold_determinant)

    print("part2 begins")
    with concurrent.futures.ThreadPoolExecutor(max_workers=11) as executor:
        MCC_futures = [executor.submit(MCC_judge, i_judge, x_judge, median_proba_threshold, median_data_to_judge) for i_judge, x_judge in itertools.product(gene_combinations[:1000], stripe_list)]
        MCC_future_results = pd.DataFrame([future.result() for future in MCC_futures])
    print("part2 done")

    if MCC_future_results["model"].all() == dataframes_from_futures["model"].all() and MCC_future_results["stripe"].all() == dataframes_from_futures["stripe"].all():
        models_final = pd.concat([dataframes_from_futures, MCC_future_results[["median_threshold", "median_MCC"]]], axis=1, sort=False)
    else:
        print("error - dataframe indexex dont match")
        exit()


    models_final.columns = ["Model", "Stripe", "Threshold", "MCC", "Coefs", "Median Threshold", "Median MCC"]
    models_final.to_csv("ftz/bdtnp_ftz_gene_discovery.csv", index=False)
    print("Start time:", time.ctime(start))
    print("Finish time:", time.ctime(time.time()))
    print(f"Run time: {(time.time() - start)/(60):.2f} minutes")