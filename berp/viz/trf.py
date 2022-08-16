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
    for i, sensor in enumerate(model.coef_):
        for j, feature in enumerate(sensor):
            for k, feature_value in enumerate(feature):
                df_data.append((i, j, k, feature_value))

    df = pd.DataFrame(df_data, columns=["sensor", "feature", "lag", "coef"])
    df["epoch_time"] = df.lag.map(dict(enumerate(model.delays_ / model.sfreq)))

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