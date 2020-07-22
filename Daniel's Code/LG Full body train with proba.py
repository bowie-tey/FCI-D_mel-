import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv('D_mel_atlas.csv')
df_1 = pd.read_csv('timepoint1.csv')
# Headless with train and test
stripe_2x = []
stripe_2z = []
stripe_2y = []
color_stripe_2 = []
color_stripe_2hb = []
color_stripe_2gt = []
color_stripe_2bcd = []
color_stripe_2kr = []

for i, x, y, z, h, g, b, k in zip(df['eve__3'], df['x__3'], df['y__3'], df['z__3'], df['hbP__3'], df['gtP__3'],
                                  df['bcdP__3'], df['KrP__3']):
    if -27 >= x >= -68 and 200 >= z >= -30 and 250 >= y >= -250 or -40 >= x >= -85 and -30 >= z >= -200:
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_stripe_2.append(int(i > 0.172))
        color_stripe_2hb.append(h)
        color_stripe_2gt.append(g)
        color_stripe_2bcd.append(b)
        color_stripe_2kr.append(k)
    else:
    #elif 250 >= x >= -27 and 200 >= z >= -30 and 250 >= y >= -250 or 250 >= x >= -40 and -30 >= z >= -200:
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_stripe_2.append(int(i > 10))  # this puts the value when its not stripe 2
        color_stripe_2hb.append(h)
        color_stripe_2gt.append(g)
        color_stripe_2bcd.append(b)
        color_stripe_2kr.append(k)

matches_1 = [i for i, x in enumerate(color_stripe_2) if x == 1]
matches_0 = [i for i, x in enumerate(color_stripe_2) if x == 0]
data_frame_comparison = pd.DataFrame(list(zip(color_stripe_2gt, color_stripe_2kr, color_stripe_2bcd, color_stripe_2hb)),
                                     columns=['Gt', 'Kr', 'Bcd', 'Hb'])
x_train, x_test, y_train, y_test = train_test_split(data_frame_comparison, color_stripe_2,
                                                  random_state=1)
###
###
###
# Logistic Regression without proba
###
###
###
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred_a = log_reg.predict(data_frame_comparison)
confusion_matrix(color_stripe_2, y_pred_a)  # TP- 2741, FP- 43, FN-46, TN-329
matthews_corrcoef(color_stripe_2, y_pred_a)  # 0.8649192349959198
# Scatter Plot
ax = plt.axes(projection='3d')
ax.set_xlim([-220, 220])  # -220,+220
ax.set_ylim([-220, 220])
ax.set_zlim([-220, 220])
ax.set_xlabel('Anteroposterior Axis', size=10)  # , size=3)
ax.set_ylabel('Lateral (Left and Right) Axis', size=10)
ax.set_zlabel('Doroventral Axis', size=10)
ax.set_title('eve', size=18)
ax.scatter(stripe_2x, stripe_2y, stripe_2z, c=y_pred_a, s=80, marker='.',
           cmap='Blues')  # stripe_2x(723 len); y_pred_a(c)
len(y_pred_a)
###
###
###
# Logistic Regression with proba
###
###
###
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg.score(x_train, y_train)
y_pred_new = log_reg.predict_proba(data_frame_comparison)
y_pred_new_log= log_reg.predict_log_proba(data_frame_comparison)
probability_1 = y_pred_new[:, 0]  # probability that the entry/sample is 1
probability_0 = y_pred_new[:, 1]  # probability that the entry/sample is 0
probability_1_log=y_pred_new_log[:,0]
probability_0_log=y_pred_new_log[:,1]


# ROC
# Roc
#roc_1 = roc_auc_score(color_stripe_2, probability_0)
#lg_fpr, lg_tpr, _ = roc_curve(color_stripe_2, probability_1)

#plt.plot(lg_fpr, lg_tpr, marker='.', )
#plt.show()

#

threshold = np.arange(0.170, 0.700, 0.001)
threshold = np.round(threshold, 3)
threshold_1 = []
thresholds_1 = {}
for z in threshold:
    for i in probability_1:
        if i < z:
            threshold_1.append(0)
        elif i > z:
            threshold_1.append(1)
    thresholds_1[z] = threshold_1
    threshold_1 = []
MCC_score_1={}
for i in thresholds_1.keys():
    MCC_score_1[i]=matthews_corrcoef(color_stripe_2, thresholds_1[i])

threshold_0 = []
thresholds_0 = {}
for z in threshold:
    for i in probability_0:
        if i < z:
            threshold_0.append(0)
        elif i > z:
            threshold_0.append(1)
    thresholds_0[z] = threshold_0
    threshold_0 = []
MCC_score_0=[]
for i in thresholds_0.keys():
    MCC_score_0.append(matthews_corrcoef(color_stripe_2, thresholds_0[i]))

plt.plot(MCC_score_0,threshold)


ax = plt.axes(projection='3d')
ax.set_xlim([-220, 220])  # -220,+220
ax.set_ylim([-220, 220])
ax.set_zlim([-220, 220])
ax.set_xlabel('Anteroposterior Axis', size=10)  # , size=3)
ax.set_ylabel('Lateral (Left and Right) Axis', size=10)
ax.set_zlabel('Doroventral Axis', size=10)
ax.set_title('eve', size=18)
ax.scatter(stripe_2x, stripe_2y, stripe_2z, c=thresholds_0[0.350], s=80, marker='.',
           cmap='Blues')  # stripe_2x(723 len); y_pred_a(c)
