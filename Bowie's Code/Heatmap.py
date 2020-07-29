import pandas as pd, matplotlib.pylab as plt, seaborn as sns
from statistics import mean

title = "4 Regulator Combination"
df = pd.read_csv("eve_tp3_MCC.csv")
CG10924 = []
CG17786 = []
CG4702 = []
Cyp310a1 = []
D_3 = []
Dfd = []
Doc2 = []
KRP = []
Traf1 = []
Brk = []
Bun = []
Cad = []
Cnc = []
Croc = []
Emc = []
Fj = []
Fkh = []
Ftz = []
GTP = []
HBP = []
h = []
Hkb = []
Kni = []
Knrl = []
Oc = []
Odd = []
Path = []
Prd = []
Rho = []
Sala = []
Slp1 = []
Slp2 = []
Sna = []
Sob = []
Srp = []
Term = []
Tll = []
Trn = []
Tsh = []
Twi = []
Zen = []
BCDP = []
#
#
keys = ['CG10924__3', 'CG17786__3', 'CG4702__3', 'Cyp310a1__3', 'D__3',
       'Dfd__3', 'Doc2__3', 'KrP__3', 'Traf1__3', 'brk__3',"bcdP__3",'bun__3', 'cad__3', 'cnc__3', 'croc__3', 'emc__3',
       'fj__3', 'fkh__3', "ftz__3",'gtP__3',"h__3", 'hbP__3', 'hkb__3', 'kni__3', 'knrl__3', 'oc__3',"odd__3" ,'path__3',"prd__3",
       'rho__3', 'sala__3', 'slp1__3', 'slp2__3', 'sna__3', 'sob__3', 'srp__3', 'term__3', 'tll__3',
       'trn__3', 'tsh__3', 'twi__3', 'zen__3']
#
alls = [CG10924, CG17786, CG4702, Cyp310a1, D_3,
       Dfd, Doc2, KRP, Traf1, Brk,BCDP, Bun, Cad, Cnc, Croc, Emc,
       Fj, Fkh,Ftz, GTP,h, HBP, Hkb, Kni, Knrl, Oc,Odd, Path,Prd,
       Rho, Sala, Slp1, Slp2, Sna, Sob, Srp, Term, Tll,
       Trn, Tsh, Twi, Zen]

df1 = pd.DataFrame(columns=keys)
df1["Regulators"] = keys
df1.set_index("Regulators")
del df1["Regulators"]



for c in df1.columns:
    for r,k in zip(df.index, keys):
        total = []
        for m,a,s in zip(df["Model"],df["MCC_081"],df["Stripe"]):
            test = [c,k]
            if a != "Nan" and s == 5:
                if all(x in m for x in test):
                    total.append(float(a))
                else:
                    None
            else:
                total.append(0)
        df1.loc[df1.index[r],c] = max(total)

print("Done")




#
# for c in df1.columns:
#     for r,k in zip(df.index, keys):
#         total = []
#         for m,a in zip(df["Models"],df["AUC"]):
#             test = [c,k]
#             if all(x in m for x in test):
#                 total.append(float(a))
#             else:
#                 None
#
#         df1.loc[df1.index[r],c] = max(total)



# print(df1)
# sns.heatmap(df1, xticklabels=keys,yticklabels=keys)
# plt.show()
# df1.to_csv("test.csv", index=False)

df = pd.read_csv("test.csv")
sns.set(font_scale=0.5)
ax = sns.heatmap(df,xticklabels=keys,yticklabels=keys,vmin=0.5, vmax=1,  linewidths=.5, cmap="YlOrRd")
ax.set_title(f"{title}",fontsize=20)
plt.show()


# MCC = []
# df = pd.read_csv("38perm2_results(3).csv")
# for i in df["MCC"]:
#     if i != "Nan":
#         MCC.append(float(i))
#     else:
#         None
# sns.distplot(MCC,axlabel="MCC Values")





