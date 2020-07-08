import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv("D_mel_atlas.csv")
columns = df.columns.to_list()

fig = plt.figure(figsize=plt.figaspect(1))  # figsize=plt.figaspect(0.5 #figsize=(5,5)
# ax= plt.axes(projection='3d')


for i in range(1, 7):
    ax = fig.add_subplot(2, 3, i, projection='3d')
    ax.set_xlim([-220, 220])  # -220,+220
    ax.set_ylim([-220, 220])
    ax.set_zlim([-220, 220])
    ax.scatter(df[f'x__{i}'], df[f'y__{i}'], df[f'z__{i}'], c=df[f'eve__{i}'], s=7, marker='.',
               cmap='viridis')  # s = 30
    ax.view_init(20, -60)  # prints the plot in a desired angle

plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)  # tightens the space between the rows and columns,
# thus making the subplots bigger
plt.show()
