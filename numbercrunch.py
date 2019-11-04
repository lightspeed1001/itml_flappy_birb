import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
birb = pd.read_csv("stats2.csv", header=None, names=["episode", "score", "type"])

sns.lineplot(data=birb, x="episode", y="score", hue="type")
plt.show()