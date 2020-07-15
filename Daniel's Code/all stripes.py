import pandas as pd
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

eve_values = df['eve'].to_list()
for i in range(0, 6077):
    if eve_values[i] >= 0.000499:  # 0.000499
        eve_values[i] = 1
    else:
        eve_values[i] = 0

stripe_1x = []
stripe_1z = []
stripe_1y = []
color_stripe_1 = []
for i, x, y, z in zip(df['eve'], df['X'], df['Y'], df['Z']):
    if -70 >= x >= -250 and 250 >= z >= -30 and 250 >= y >= -250 or -80 >= x >= -250 and -30 >= z >= -250:
        stripe_1x.append(x)
        stripe_1z.append(z)
        stripe_1y.append(y)
        color_stripe_1.append(i)

for i in range(0, len(color_stripe_1)):
    if color_stripe_1[i] >= 0.000499:  # 0.000499
        color_stripe_1[i] = 0.1
    else:
        color_stripe_1[i] = 0.5

stripe_2x = []
stripe_2z = []
stripe_2y = []
color_stripe_2 = []
for i, x, y, z in zip(df['eve'], df['X'], df['Y'], df['Z']):
    if -27 >= x >= -70 and 200 >= z >= -30 and 250 >= y >= -250 or -40 >= x >= -80 and -30 >= z >= -200:
        stripe_2x.append(x)
        stripe_2z.append(z)
        stripe_2y.append(y)
        color_stripe_2.append(i)

for i in range(0, len(color_stripe_2)):
    if color_stripe_2[i] >= 0.000499:  # 0.000499
        color_stripe_2[i] = 0.2
    else:
        color_stripe_2[i] = 0.5

stripe_3x = []
stripe_3z = []
stripe_3y = []
color_stripe_3 = []
for i, x, y, z in zip(df['eve'], df['X'], df['Y'], df['Z']):
    if 0 >= x >= -27 and 200 >= z >= -30 and 250 >= y >= -250 or -10 >= x >= -40 and -30 >= z >= -200:
        stripe_3x.append(x)
        stripe_3z.append(z)
        stripe_3y.append(y)
        color_stripe_3.append(i)

for i in range(0, len(color_stripe_3)):
    if color_stripe_3[i] >= 0.000499:  # 0.000499
        color_stripe_3[i] = 0.3
    else:
        color_stripe_3[i] = 0.5

stripe_4x = []
stripe_4z = []
stripe_4y = []
color_stripe_4 = []
for i, x, y, z in zip(df['eve'], df['X'], df['Y'], df['Z']):
    if 30 >= x >= 0 and 200 >= z >= -30 and 250 >= y >= -250 or 30 >= x >= -10 and -30 >= z >= -200:
        stripe_4x.append(x)
        stripe_4z.append(z)
        stripe_4y.append(y)
        color_stripe_4.append(i)

for i in range(0, len(color_stripe_4)):
    if color_stripe_4[i] >= 0.000499:  # 0.000499
        color_stripe_4[i] = 0.6
    else:
        color_stripe_4[i] = 0.5

stripe_5x = []
stripe_5z = []
stripe_5y = []
color_stripe_5 = []
for i, x, y, z in zip(df['eve'], df['X'], df['Y'], df['Z']):
    if 50 >= x >= 30 and 200 >= z >= -30 and 250 >= y >= -250 or 70 >= x >= 30 and -30 >= z >= -200:
        stripe_5x.append(x)
        stripe_5z.append(z)
        stripe_5y.append(y)
        color_stripe_5.append(i)

for i in range(0, len(color_stripe_5)):
    if color_stripe_5[i] >= 0.000499:  # 0.000499
        color_stripe_5[i] = 0.7
    else:
        color_stripe_5[i] = 0.5

stripe_6x = []
stripe_6z = []
stripe_6y = []
color_stripe_6 = []
for i, x, y, z in zip(df['eve'], df['X'], df['Y'], df['Z']):
    if 81 >= x >= 50 and 200 >= z >= -10 and 250 >= y >= -250 or 110 >= x >= 70 and -30 >= z >= -200 or -10 >= z >= -30 and 100 >= x >= 50:
        stripe_6x.append(x)
        stripe_6z.append(z)
        stripe_6y.append(y)
        color_stripe_6.append(i)

for i in range(0, len(color_stripe_6)):
    if color_stripe_6[i] >= 0.000499:  # 0.000499
        color_stripe_6[i] = 0.8
    else:
        color_stripe_6[i] = 0.5

stripe_7x = []
stripe_7z = []
stripe_7y = []
color_stripe_7 = []
for i, x, y, z in zip(df['eve'], df['X'], df['Y'], df['Z']):
    if 250 >= x >= 81 and 200 >= z >= -10 and 250 >= y >= -250 or 250 >= x >= 110 and -30 >= z >= -200 or -10 >= z >= -30 and 250 >= x >= 100:
        stripe_7x.append(x)
        stripe_7z.append(z)
        stripe_7y.append(y)
        color_stripe_7.append(i)

for i in range(0, len(color_stripe_7)):
    if color_stripe_7[i] >= 0.000499:  # 0.000499
        color_stripe_7[i] = 0.9
    else:
        color_stripe_7[i] = 0.5

# All stripes
# for j, i in zip(gene, range(1, 44)):
ax = plt.axes(projection='3d')  # fig.add_subplot(7, 7, i, projection="rectilinear")
ax.set_xlim([-220, 220])  # -220,+220
ax.set_ylim([-220, 220])
ax.set_zlim([-220, 220])
ax.set_xlabel('Anteroposterior Axis', size=10)  # , size=3)
ax.set_ylabel('Dorsoventral Axis', size=10)
ax.set_title('eve', size=18)  # f'{j}'

ax.scatter(stripe_1x, stripe_1y, stripe_1z, c=color_stripe_1, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)  # s = 30
ax.scatter(stripe_2x, stripe_2y, stripe_2z, c=color_stripe_2, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)
ax.scatter(stripe_3x, stripe_3y, stripe_3z, c=color_stripe_3, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)
ax.scatter(stripe_4x, stripe_4y, stripe_4z, c=color_stripe_4, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)  # s = 30
ax.scatter(stripe_5x, stripe_5y, stripe_5z, c=color_stripe_5, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)  # s = 30
ax.scatter(stripe_6x, stripe_6y, stripe_6z, c=color_stripe_6, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)
ax.scatter(stripe_7x, stripe_7y, stripe_7z, c=color_stripe_7, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)
plt.show()

# Test for each stripe independently
# for j, i in zip(gene, range(1, 44)):
ax = plt.axes(projection='3d')  # fig.add_subplot(7, 7, i, projection="rectilinear")
ax.set_xlim([-220, 220])  # -220,+220
ax.set_ylim([-220, 220])
ax.set_zlim([-220, 220])
ax.set_xlabel('Anteroposterior Axis', size=10)  # , size=3)
ax.set_ylabel('Dorsoventral Axis', size=10)
ax.set_title('eve', size=18)  # f'{j}'

ax.scatter(stripe_7x, stripe_7y, stripe_7z, c=color_stripe_7, s=80, marker='.',
           # s= 7 # c=j #gene['eve']
           cmap='hsv', vmin=0, vmax=1)  # s = 30
# ax.set_yticklabels([])
# ax.set_xticklabels([])
# plt.savefig(f'{j}_dge.png')
