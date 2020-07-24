import pandas as pd
import seaborn as sns
from pylab import *  # for subplot
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('final_dge_normal.csv')
df_short = pd.read_csv('timepoint_3.csv')

genes = {}
header = df.columns.to_list()

header_short = df_short.columns.to_list()

j = 0
for i in header_short:
    header_short[j] = i.strip('__3')
    j += 1
    header_short[0] = 'X'
    header_short[1] = 'Y'
    header_short[2] = 'Z'

for i in header:
    for j in header_short:
        if j == i:
            genes[i] = df[i].to_list()

gene = pd.DataFrame.from_dict(genes)
gene.to_csv('gene_plot.csv', index=False)
columns = gene.columns.to_list()

# Here we basically remove the genes from data set 3 that match the genes from the new data set and make a new dataframe with them
k = 0
for i in columns:
    columns[k] = i + '__3'
    k += 1
    columns[-3] = 'x__3'
    columns[-2] = 'y__3'
    columns[-1] = 'z__3'

genes_3 = {}
header_short = df_short.columns.to_list()
for i in columns:
    for j in header_short:
        if j == i:
            genes_3[j] = df_short[j].to_list()

gene_3 = pd.DataFrame.from_dict(genes_3)
gene_3.to_csv('gene_plot_3.csv', index=False)

# sns.set(style='whitegrid',color_codes=True)
# sns.distplot(gene)

# fig = plt.figure(figsize=plt.figaspect(1))
# ax = fig.add_subplot(7, 7, 1, prjection='3d')
# ax.sns.plot(gene['eve'])

# Plot all the genes from the modified 3 point data

sns.set(rc={'figure.figsize': (16, 8)})
for i, j in zip(gene_3, range(1, 44)):
    subplot(7, 7, j)
    ax = sns.distplot(df_short[i])
plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)