import pandas as pd
from scipy.stats import ttest_1samp

ep = pd.read_csv("out/episodes.csv")
print(ep.groupby("controller")[["mean_delay_s", "amb_delay_s"]].mean().round(1))
print()

w = ep.pivot(index="seed", columns="controller")
for metric in ["mean_delay_s", "amb_delay_s"]:
    for c in ["longest_queue", "priority"]:
        diff = w[metric][c] - w[metric]["fixed"]
        t, p = ttest_1samp(diff, 0)
        print(f"{metric:13s} {c:14s} mean diff vs fixed = {diff.mean():6.1f} s   p = {p:.3f}")