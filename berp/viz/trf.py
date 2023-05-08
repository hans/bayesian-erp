"""
Tools for visualizing fitted TRF models.
"""

from typing import *

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from berp.models.trf import TemporalReceptiveField


def trf_to_dataframe(model: TemporalReceptiveField,
                     predictor_names: Optional[List[str]] = None) -> pd.DataFrame:
    df_data = []
    for i, feature in enumerate(model.coef_.detach().cpu().numpy()):
        for j, lag in enumerate(feature):
            for k, sensor_coef in enumerate(lag):
                df_data.append((i, j, k, sensor_coef))

    df = pd.DataFrame(df_data, columns=["predictor", "lag", "sensor", "coef"])
    df["epoch_time"] = df.lag.map(dict(enumerate(model.delays_.numpy() / model.sfreq)))

    if predictor_names is not None:
        df["predictor_name"] = df.predictor.map(dict(enumerate(predictor_names)))

    return df


def plot_trf_coefficients(model_or_df: Union[TemporalReceptiveField, pd.DataFrame],
                          errorbar=("ci", 95),
                          predictor_names=None,
                          predictor_match_patterns=None,
                          plot_grids=True,
                          **kwargs) -> plt.Figure:
    if isinstance(model_or_df, TemporalReceptiveField):
        df = trf_to_dataframe(model_or_df, predictor_names=predictor_names)
    elif isinstance(model_or_df, pd.DataFrame):
        df = model_or_df
    else:
        raise ValueError("Model or dataframe expected")

    if predictor_match_patterns is not None and "predictor_name" not in df.columns:
        raise ValueError("predictor patterns provided, but no predictor names")

    if predictor_match_patterns is not None:
        to_plot = df[df.predictor_name.str.contains("|".join(predictor_match_patterns), regex=True)]
    else:
        to_plot = df[:]

    hue = "predictor_name" if predictor_names is not None else "predictor"
    ax = sns.lineplot(data=to_plot.reset_index(),
                      x="epoch_time", y="coef", hue="predictor_name",
                      errorbar=errorbar, **kwargs)

    ax.set_xlabel("Epoch time")
    ax.set_ylabel("TRF coefficient")

    if plot_grids:
        ax.axhline(0, c="gray", alpha=0.3)
        ax.axvline(0, c="gray", alpha=0.3)
        ax.axvline(0.3, c="gray", alpha=0.3, linestyle="dashed")
        ax.axvline(0.5, c="gray", alpha=0.3, linestyle="dashed")

    plt.legend(loc=(1.05, 0.1))

    # plt.subplots_adjust(top=0.9)
    # plt.suptitle(f"TRF weights, avg over {len(to_plot.subject.unique())} subjects $\\times$ {len(to_plot.split.unique())} CV folds")
    plt.tight_layout()

    return plt.gcf()
