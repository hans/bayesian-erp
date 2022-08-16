"""
Tools for visualizing fitted TRF models.
"""

from typing import *

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from berp.models.trf import TemporalReceptiveField


def trf_to_dataframe(model: TemporalReceptiveField) -> pd.DataFrame:
    df_data = []
    for i, feature in enumerate(model.coef_.detach().numpy()):
        for j, lag in enumerate(feature):
            for k, sensor_coef in enumerate(lag):
                df_data.append((i, j, k, sensor_coef))

    df = pd.DataFrame(df_data, columns=["feature", "lag", "sensor", "coef"])
    df["epoch_time"] = df.lag.map(dict(enumerate(model.delays_.numpy() / model.sfreq)))

    return df


def plot_trf_coefficients(model: TemporalReceptiveField, ci=95, **kwargs) -> plt.Figure:
    df = trf_to_dataframe(model)

    # TODO feature subsetting
    to_plot = df[:]

    ax = sns.lineplot(data=to_plot.reset_index(),
                      x="epoch_time", y="coef", hue="feature",
                      ci=ci, **kwargs)

    ax.set_xlabel("Epoch time")
    ax.set_ylabel("TRF coefficient")
    
    ax.axhline(0, c="gray", alpha=0.3)
    ax.axvline(0, c="gray", alpha=0.3)
    ax.axvline(0.3, c="gray", alpha=0.3, linestyle="dashed")
    ax.axvline(0.5, c="gray", alpha=0.3, linestyle="dashed")
        
    plt.subplots_adjust(top=0.9)
    # plt.suptitle(f"TRF weights, avg over {len(to_plot.subject.unique())} subjects $\\times$ {len(to_plot.split.unique())} CV folds")
    plt.tight_layout()
        
    return plt.gcf()