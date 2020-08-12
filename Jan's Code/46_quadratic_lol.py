import pandas as pd
import numpy as np
import itertools
from imblearn.over_sampling import SMOTE
import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

###########FUNCTIONS
###########FUNCTIONS
def threshold_predict_proba(predict_proba, proba_threshold):
    predict_onoff = predict_proba[:, 1] >= proba_threshold
    predict_onoff.astype(int)
    return predict_onoff

def match_mismatch(predict_onoff, test):
    match_mismatch = np.array([])

    # counting the number of false pos, false neg, and true matches
    for predict, tes in zip(predict_onoff, test):
        if predict != tes and tes == True:
            match_mismatch = np.append(match_mismatch, "false negative")
        elif predict != tes and tes == False:
            match_mismatch = np.append(match_mismatch, "false positive")
        elif predict == tes and tes == True:
            match_mismatch = np.append(match_mismatch, "match positive")
        elif predict == tes and tes == False:
            match_mismatch = np.append(match_mismatch, "match negative")
        else:
            raise Exception("match_mismatch - ERROR - script out of bounds")

    return match_mismatch
###########FUNCTIONS
###########FUNCTIONS

#dataset
dataset = pd.read_csv("D_mel_atlas.csv")
stripe_identity = pd.read_csv("Jans_stripes_from_bdtnp2.csv")

##GENES OF INTEREST: kni; tll; hb; Kr; D; gt
genes_of_interest = ["kni__3", "tll__3", "hbP__3", "KrP__3", "D__3", "gtP__3"]
combinations = list(itertools.combinations(genes_of_interest, 5))

eve_in = stripe_identity["stripe_4+6"].to_numpy()
for combination in combinations:
    combination = list(combination)

    eve_ex = dataset[combination].to_numpy()
    for gene in combination:
        quad = dataset[gene].to_numpy() ** 2
        eve_ex = np.append(eve_ex, quad.reshape(len(quad),1), axis=1)

        # balancing
        ftz_ex_train, ftz_ex_test, ftz_in_train, ftz_in_test = train_test_split(eve_ex, eve_in, test_size=0.3, random_state=0)
        os = SMOTE(random_state=0)
        ftz_ex_os, ftz_in_os = os.fit_sample(ftz_ex_train, ftz_in_train)

        # model
        model = LogisticRegression(C=1e10, max_iter=1000)
        model.fit(ftz_ex_os, ftz_in_os)

        # predict
        predict_proba = model.predict_proba(eve_ex)

        # decide whether the cell is eve on/off based on the prediction and calculate MCC for the prediction
        threshold = 0.81
        predict_onoff = threshold_predict_proba(predict_proba, threshold)
        MCC = sklearn.metrics.matthews_corrcoef(eve_in, predict_onoff)

        if MCC > 0.7:
            match_identity = match_mismatch(predict_onoff, eve_in)

            # making color array for plotting below
            colors = np.array([])
            for i in match_identity:
                if i == "false positive":
                    colors = np.append(colors, "#f130ff")  # purple
                elif i == "false negative":
                    colors = np.append(colors, "#ff1900")  # red
                elif i == "match positive":
                    colors = np.append(colors, "#0062ff")  # blue
                elif i == "match negative":
                    colors = np.append(colors, "#fafafa")  # white

            # saving the number of false positives, negatives, true positives and negatives
            fp = (match_identity == "false positive").sum()
            fn = (match_identity == "false negative").sum()
            tp = (stripe_identity[f"stripe_4+6"] == 1).sum()
            tn = (stripe_identity[f"stripe_4+6"] == 0).sum()

            ###PLOTING
            plot = plt.axes(projection="3d")
            plot.set_xlim([-220, 220])
            plot.set_ylim([-220, 220])
            plot.set_zlim([-220, 220])
            plot.view_init(elev=10, azim=-90)
            plot.set_title(f"{combination}+quad: {gene}; MCC:{MCC:.2f}")
            plot.scatter(dataset["x__3"].tolist(), dataset["y__3"].tolist(), dataset["z__3"].tolist(), c=colors.tolist(), s=5, marker=".")
            plt.savefig(f"lol/5/{combination}_quad_{gene}.png")
            plt.close

