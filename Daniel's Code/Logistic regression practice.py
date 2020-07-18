# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0).fit(X, y)
# clf.predict(X[:2, :])
# array([0, 0])
# clf.predict_proba(X[:2, :])
# array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
#       [9.7...e-01, 2.8...e-02, ...e-08]])
# clf.score(X, y)
# 0.97...
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

df = pd.read_csv('D_mel_atlas.csv')
df_1 = pd.read_csv('timepoint1.csv')

# eve_values = df['eve__3'].to_list()
# for i in range(0,6077):
#    if eve_values[i] >= 0.172:
#       eve_values[i] = 1
#    else:
#       eve_values[i] = 0
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
        color_stripe_2.append(int(i > 0.172))
        color_stripe_2hb.append(h)
        color_stripe_2gt.append(g)
        color_stripe_2bcd.append(b)
        color_stripe_2kr.append(k)

# for i in range(722,6077):
#    color_stripe_2.append(0)


data_frame_comparison = pd.DataFrame(list(zip(color_stripe_2gt, color_stripe_2kr, color_stripe_2bcd, color_stripe_2hb)),
                                     columns=['Gt', 'Kr', 'Bcd', 'Hb'])

x_train, x_test, y_train, y_test = train_test_split(data_frame_comparison, color_stripe_2, random_state=1)
print(x_train.shape)
print(log_reg.coef_)  # 4 slopes
print(log_ref.intercept_)  # 1 intercept
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
confusion_matrix(y_test, y_pred)  # TP- 77, FP- 17, FN-3, TN-84
matthews_corrcoef(y_test, y_pred)  # 0.7894103311696256
