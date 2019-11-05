import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
birb = pd.read_csv("18k_decreasing_epsilon.csv")#, header=None, names=["episode", "score", "type"])
birb = birb.drop(birb[birb.type == "train"].index)
sns.scatterplot(data=birb, x="episode", y="score")
plt.show()