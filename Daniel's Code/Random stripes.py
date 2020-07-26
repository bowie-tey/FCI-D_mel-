import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
import collections
from imblearn.over_sampling import SMOTE  # oversampling
import collections  # count the different numbers inside a list
from imblearn.pipeline import Pipeline  # makes a function to be fitted
from imblearn.under_sampling import RandomUnderSampler  # undersampling
import itertools

df = pd.read_csv('D_mel_atlas.csv')
df_1 = pd.read_csv('timepoint1.csv')

time_point_3_genes = {}
header = df.columns.to_list()
for i in header:
    if '__3' in i and 'x__3' not in i and 'y__3' not in i and 'z__3' not in i:
        time_point_3_genes[i] = df[i].to_list()
g3 = pd.DataFrame(time_point_3_genes)
stripe_1x = []
stripe_1z = []
stripe_1y = []
color_horizont = []

# Horizontal: Model 1
for x, y, z in zip(df['x__3'], df['y__3'], df['z__3']):
    if 250 >= x >= -250 and 20 >= z >= -20 and 150 >= y >= -150:  # horizontal
        stripe_1x.append(x)
        stripe_1y.append(y)
        stripe_1z.append(z)
        color_horizont.append(1)


    elif 150 >= y >= -150 and 90 >= z >= 55 and 250 >= x >= -250:  # horizntal
        stripe_1x.append(x)
        stripe_1y.append(y)
        stripe_1z.append(z)
        color_horizont.append(1)

    elif 150 >= y >= -150 and -55 >= z >= -90 and 250 >= x >= -250:  # horizntal
        stripe_1x.append(x)
        stripe_1y.append(y)
        stripe_1z.append(z)
        color_horizont.append(1)

    else:
        stripe_1x.append(x)
        stripe_1y.append(y)
        stripe_1z.append(z)
        color_horizont.append(0)

Model_1_horizontal = pd.DataFrame(list(zip(stripe_1x, stripe_1y, stripe_1z, color_horizont)),
                                  columns=['X', 'Y', 'Z', 'Color'])
# The 45 genes order by stripes
gene_r = {}
gene_l = []
for g in g3.columns.to_list():
    for x, y, z, i in zip(df['x__3'], df['y__3'], df['z__3'], g3[f'{g}']):
        if 250 >= x >= -250 and 20 >= z >= -20 and 150 >= y >= -150:  # horizontal
            gene_l.append(i)
        elif 150 >= y >= -150 and 90 >= z >= 55 and 250 >= x >= -250:  # horizntal
            gene_l.append(i)
        elif 150 >= y >= -150 and -55 >= z >= -90 and 250 >= x >= -250:  # horizntal
            gene_l.append(i)

        else:
            gene_l.append(i)
    gene_r[g] = gene_l
    gene_l = []
all_genes = pd.DataFrame(gene_r)

full_df = pd.concat([Model_1_horizontal, all_genes], axis=1)

ax = plt.axes(projection='3d')
ax.set_xlim([-220, 220])  # -220,+220
ax.set_ylim([-220, 220])
ax.set_zlim([-220, 220])
ax.set_xlabel('Anteroposterior Axis', size=10)  # , size=3)
ax.set_ylabel('Lateral (Left and Right) Axis', size=10)
ax.set_zlabel('Doroventral Axis', size=10)
ax.set_title('eve', size=18)
ax.scatter(stripe_1x, stripe_1y, stripe_1z, c=color_horizont, s=80, marker='.',
           cmap='Blues')  # stripe_2x(723 len); y_pred_a(c)

# Vertical: Model 2
stripe_2x = []
stripe_2z = []
stripe_2y = []
color_vertical = []

for x, y, z in zip(df['x__3'], df['y__3'], df['z__3']):
    if 250 >= y >= -250 and -25 >= z >= -200 and -60 >= x >= -90:  # left bot
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)

    elif 250 >= y >= -250 and 40 >= z >= -25 and -65 >= x >= -85:  # left mid
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)

    elif 250 >= y >= -250 and -25 >= z >= -200 and 90 >= x >= 60:  # right bot
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)
    elif 250 >= y >= -250 and 40 >= z >= -25 and 85 >= x >= 65:  # right mid
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)

    elif 250 >= y >= -250 and 200 >= z >= -200 and 20 >= x >= -20:  # middle
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)
    elif 250 >= y >= -250 and 200 >= z >= 40 and 90 >= x >= 60:  # top right
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)
    elif 250 >= y >= -250 and 200 >= z >= 40 and -60 >= x >= -90:  # top left
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)
    elif 250 >= y >= -250 and 200 >= z >= -200 and -120 >= x >= -150:  # left 2nd
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)
    elif 250 >= y >= -250 and 200 >= z >= -200 and 150 >= x >= 120:  # right 2nd
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)
    elif 250 >= y >= -250 and 200 >= z >= -200 and -180 >= x >= -250:  # left tip
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)
    elif 250 >= y >= -250 and 200 >= z >= -200 and 250 >= x >= 180:  # right tip
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(1)

    else:
        # elif 250 >= x >= -27 and 200 >= z >= -30 and 250 >= y >= -250 or 250 >= x >= -40 and -30 >= z >= -200:
        stripe_2x.append(x)
        stripe_2y.append(y)
        stripe_2z.append(z)
        color_vertical.append(0)  # t

Model_2_vertical = pd.DataFrame(list(zip(stripe_2x, stripe_2y, stripe_2z, color_vertical)),
                                columns=['X', 'Y', 'Z', 'Color'])

# All 45 genes
gene_r_2 = {}
gene_l_2 = []
for g in g3.columns.to_list():
    for x, y, z, i in zip(df['x__3'], df['y__3'], df['z__3'], g3[f'{g}']):
        if 250 >= y >= -250 and -25 >= z >= -200 and -60 >= x >= -90:  # left bot
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 40 >= z >= -25 and -65 >= x >= -85:  # left mid
            gene_l_2.append(i)
        elif 250 >= y >= -250 and -25 >= z >= -200 and 90 >= x >= 60:  # right bot
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 40 >= z >= -25 and 85 >= x >= 65:  # right mid
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 200 >= z >= -200 and 20 >= x >= -20:  # middle
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 200 >= z >= 40 and 90 >= x >= 60:  # top right
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 200 >= z >= 40 and -60 >= x >= -90:  # top left
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 200 >= z >= -200 and -120 >= x >= -150:  # left 2nd
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 200 >= z >= -200 and 150 >= x >= 120:  # right 2nd
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 200 >= z >= -200 and -180 >= x >= -250:  # left tip
            gene_l_2.append(i)
        elif 250 >= y >= -250 and 200 >= z >= -200 and 250 >= x >= 180:  # right tip
            gene_l_2.append(i)
        else:
            gene_l_2.append(i)
    gene_r_2[g] = gene_l_2
    gene_l_2 = []

all_genes_2 = pd.DataFrame(gene_r_2)

full_2_df = pd.concat([Model_2_vertical, all_genes_2], axis=1)

ax = plt.axes(projection='3d')
ax.set_xlim([-220, 220])  # -220,+220
ax.set_ylim([-220, 220])
ax.set_zlim([-220, 220])
ax.set_xlabel('Anteroposterior Axis', size=10)  # , size=3)
ax.set_ylabel('Lateral (Left and Right) Axis', size=10)
ax.set_zlabel('Doroventral Axis', size=10)
ax.set_title('eve', size=18)
ax.scatter(stripe_2x, stripe_2y, stripe_2z, c=color_vertical, s=80, marker='.',
           cmap='Blues')  # stripe_2x(723 len); y_pred_a(c)

# Gene Discovery
# mains : full_df and full_2_df  all_genes and all_genes_2

# bills = [20, 20, 20, 10, 10, 10, 10, 10, 5, 5, 1, 1, 1, 1, 1]
# kur = {'p': 1, 'k': 2}
# riba = list(itertools.combinations(bills, 3))  # first argument is the list of numbers we want to make combinations with
# the second argument is the

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(all_genes, color_horizont,
                                                    random_state=1)
# Balance the train set
oversample = SMOTE(random_state=0)
x_train, y_train = oversample.fit_resample(x_train, y_train)

horizontal_gene_combinations_x = list(itertools.combinations(x_train.columns, 4))
# n_list=range(0,6078)
# data_frame_comparison= pd.concat([df_comparison,all_genes[horizontal_gene_combinations[0][0]]],axis=1)


####
####
####
# Version 1
df_comparison = pd.DataFrame()
df_dict = {}

x_train_set = pd.DataFrame()
y_train_set = pd.DataFrame()
for r, i in zip(range(0, 148995), horizontal_gene_combinations_x):
    for x in i:
         df_comparison = pd.concat([df_comparison, x_train[x]], axis=1)
         if len(df_comparison.columns) == 4:
            df_dict[r] = df_comparison
            df_comparison = pd.DataFrame()
             # 40 min yes

# Version 2
df_dict = {}
for i in range(148995):
    gene_col=pd.Series(horizontal_gene_combinations_x[i])
    df_dict[i]=x_train[gene_col]

# Version 3
df_comparison = pd.DataFrame()
df_dict = {}
x_train_set = pd.DataFrame()
y_train_set = pd.DataFrame()
for r, i in zip(range(0, 148994), horizontal_gene_combinations_x):
    for x in i:
         df_comparison = pd.concat([df_comparison, x_train[x]], axis=1)
         if len(df_comparison.columns) == 3:
          log_reg = LogisticRegression(class_weight='balanced')
          log_reg.fit(x_train, y_train)
          y_predict = log_reg.predict_proba(all_genes)
          probability_1 = y_predict[:, 1]  # probability that the entry/sample is 1
          threshold_1=[]
             if probability_1<0.5:
                 threshold_1.append(1)
             else:
                 threshold_1.append(0)
              MCC_score_1.append(matthews_corrcoef(color_horizont, thresholds_1))



          df_comparison = pd.DataFrame()




####
####
####


