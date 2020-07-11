import pandas as pd
import seaborn as sns
from pylab import *  # for subplot
import matplotlib.pyplot as plt

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
####### 3D
# fig = plt.figure(figsize=plt.figaspect(1))


for z,j, i in zip(gene,gene_3, range(1, 44)):
    ax = plt.axes(projection='rectilinear')  # fig.add_subplot(7, 7, i, projection="rectilinear")
    ax.set_xlim([-220, 220])  # -220,+220
    ax.set_ylim([-220, 220])
    ax.set_xlabel('Anteroposterior Axis', size=10)  # , size=3)
    ax.set_ylabel('Dorsoventral Axis', size=10)
    ax.set_title(f'{z}',size=18)  # f'{j}'
    # ax.set_zlim([-220, 220])
    ax.scatter(gene_3['x__3'], gene_3['z__3'], c=gene_3[j], s=40, marker='.',  # s= 7 # c=j
               cmap='Blues',vmin = 0, vmax = 1)  # s = 30
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.savefig(f'{z}_bdtnp.png')
# ax.view_init(20, -60)  # prints the plot in a desired angle

# plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # tightens the space between the rows and columns,
# thus making the subplots bigger

#plt.show()
