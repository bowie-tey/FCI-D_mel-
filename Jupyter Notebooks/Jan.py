import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

dataset = pd.read_csv("D_mel_atlas.csv")
#column_list = dataset.columns.to_list()

#eve_expression = {}
for x in range(6):
    plot = plt.axes(projection="3d")
    plot.set_xlim([-220,220])
    plot.set_ylim([-220, 220])
    plot.set_zlim([-220, 220])
    plot.set_title("eve__{0}".format(x+1))
    plot.set_xticks([])
    plot.set_yticks([])
    plot.set_zticks([])
    plot.view_init(elev=10., azim=90)
    plot.scatter(dataset["x__{0}".format(x+1)], dataset["y__{0}".format(x+1)], dataset["z__{0}".format(x+1)], c=dataset["eve__{0}".format(x+1)], s=5, marker=".", cmap = "Blues")
    plt.savefig("eve__{0}".format(x+1), dpi=600)
    plt.close
