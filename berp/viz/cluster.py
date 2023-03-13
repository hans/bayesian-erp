"""
Statistical tools and visualization tools for spatiotemporal cluster
analysis.
"""

from typing import List, Tuple, NamedTuple, Union

import matplotlib.pyplot as plt
import mne
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from berp.viz.trf import plot_trf_coefficients


class ClusterResult(NamedTuple):
    T_obs: np.ndarray
    clusters: List[Tuple[np.ndarray, np.ndarray]]
    cluster_p_values: np.ndarray
    H0: np.ndarray

    info: mne.Info


def get_coef_ndarray(trf_df: pd.DataFrame) -> np.ndarray:
    """
    returns n_subjects * n_times * n_channels coefficient representation for
    statistical analysis
    """
    ret = trf_df.reset_index().set_index(["subject", "epoch_time", "sensor"]).coef
    subjects = ret.index.unique(level="subject")
    times = ret.index.unique(level="epoch_time")
    sensors = ret.index.unique(level="sensor")
    
    return ret.to_numpy() \
        .reshape((len(subjects), len(times), len(sensors)))


def cluster_predictor(predictor_df: pd.DataFrame, info=None, adjacency=None,
                      plot=True, **cluster_kwargs) -> ClusterResult:
    coef_estimates = get_coef_ndarray(predictor_df)
    
    result = ClusterResult(
        *mne.stats.spatio_temporal_cluster_1samp_test(
            coef_estimates, adjacency=adjacency,
            **cluster_kwargs),
        info=info)
    
    if plot:
        plot_cluster_result(result, predictor_df)
    
    return result


def plot_cluster_result(result: ClusterResult, trf_sub_df: pd.DataFrame):
    good_cluster_idxs = np.where(result.cluster_p_values < 0.05)[0]
    
    for i_clu, clu_idx in enumerate(tqdm(good_cluster_idxs)):
        time_inds, space_inds = np.squeeze(result.clusters[clu_idx])

        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for T stat
        t_map = result.T_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = time_inds / result.info['sfreq']

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], result.info, tmin=0)
        t_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo, cmap='Reds',
                              vlim=(np.min, np.max), show=False,
                              colorbar=False, mask_params=dict(markersize=10))
        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            'Averaged T-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += "s (mean)"
        
        to_plot = trf_sub_df[trf_sub_df.sensor.isin(ch_inds)]
        plot_trf_coefficients(to_plot, errorbar="se", ax=ax_signals)
        ax_signals.set_title(title)
        ax_signals.get_legend().remove()

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                 color='orange', alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        plt.show()
        
    return plt.gcf()