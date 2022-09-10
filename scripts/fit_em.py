import json
from pathlib import Path
import pickle

import hydra
from omegaconf import OmegaConf
import optuna
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm

from berp.config import Config, CVConfig
from berp.cv import OptunaSearchCV
from berp.models.trf_em import GroupBerpTRFForwardPipeline, GroupTRFForwardPipeline
from berp.viz.trf import trf_to_dataframe, plot_trf_coefficients


def make_cv(model, cfg: CVConfig):
    """
    Make cross-validation object.
    """
    # TODO if we want to customize sampler, we have to set up a "study"
    param_sampler = hydra.utils.instantiate(cfg.param_sampler)
    param_distributions = {}
    for name, dist_cfg in cfg.params.items():
        param_distributions.update(hydra.utils.call(dist_cfg, name=name))
    
    sampler = optuna.samplers.TPESampler(multivariate=True)
    study = optuna.create_study(sampler=sampler, direction="maximize")
    
    return OptunaSearchCV(
        estimator=clone(model),
        study=study,
        enable_pruning=True,
        max_iter=10, n_trials=20,
        param_distributions=param_distributions,
        error_score="raise",
        cv=KFold(n_splits=cfg.n_inner_folds, shuffle=False),
        refit=True,
        verbose=1,)


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: Config):
    print(OmegaConf.to_yaml(cfg))

    dataset = hydra.utils.call(cfg.dataset)
    dataset.set_n_splits(4)

    model = hydra.utils.call(cfg.model,
                             n_outputs=dataset.n_sensors,
                             n_phonemes=dataset.n_phonemes,
                             optim=cfg.solver)
    from pprint import pprint; pprint(model.get_params())
    # model.partial_fit(dataset)
    # model.set_params(trf__alpha=np.ones(129))
    # nbd = NestedBerpDataset([dataset.datasets[0]], n_splits=4)
    # model.partial_fit(nbd)
    # model.partial_fit(nbd)
    # import ipdb; ipdb.set_trace()
    # return

    # Before splitting datasets, prime model pipeline with full data.
    model.prime(dataset)

    data_train, data_test = train_test_split(dataset, test_size=0.25, shuffle=False)
    cv = make_cv(model, cfg.cv)
    cv.fit(data_train)

    # Save study information for all hparam options.
    cv.study.trials_dataframe().to_csv("trials.csv", index=False)

    params_dir = Path("params")
    params_dir.mkdir()
    np.savez(params_dir / "hparams.npz", **cv.best_params_)
    est = cv.best_estimator_

    # May have a wrapper around it.
    if hasattr(est, "pipeline"):
        est = est.pipeline
    assert isinstance(est, GroupTRFForwardPipeline)

    # Save the whole pipeline first in pickle format. This includes
    # latent model parameters as well as all encoder weights. It does
    # not include pre-transformed TRF design matrices
    with (params_dir / "pipeline.pkl").open("wb") as f:
        pickle.dump(est, f)

    # Extract and save particular parameters from the ModelParameters grid.
    if hasattr(est, "params"):
        # TODO should just be linked to the ones we know we are searching over
        berp_params_to_extract = ["threshold", "lambda_"]
        berp_params_df = pd.DataFrame(
            [[getattr(param_set, param_name).item() for param_name in berp_params_to_extract]
            for param_set in est.params],
            columns=berp_params_to_extract
        )
        berp_params_df["weight"] = est.param_weights.numpy()
        berp_params_df.to_csv(params_dir / "berp_params.csv", index=False)

    # Table-ize and render TRF coefficients.
    ts_feature_names = dataset.ts_feature_names if dataset.ts_feature_names is not None else \
        [str(x) for x in range(dataset.n_ts_features)]
    variable_feature_names = dataset.variable_feature_names if dataset.variable_feature_names is not None else \
        [f"var_{x}" for x in range(dataset.n_variable_features)]
    feature_names = ts_feature_names + variable_feature_names
    for key, encoder in tqdm(est.encoders_.items(), desc="Visualizing encoders"):
        coefs_df = trf_to_dataframe(encoder, feature_names=feature_names)
        coefs_df["name"] = key
        coefs_df.to_csv(params_dir / f"encoder_coefs.{key}.csv", index=False)

        fig = plot_trf_coefficients(
            encoder,
            feature_names=feature_names,
            feature_match_patterns=cfg.viz.feature_patterns)
        fig.savefig(params_dir / f"encoder_coefs.{key}.png")

    # TODO calculate score on test set. or do full CV.

    # # TODO do we need to clear cache?
    # for dataset in tqdm(dataset.datasets, desc="Datasets"):
    #     data_train = NestedBerpDataset([dataset], n_splits=4)

    #     cv = make_cv(model, cfg.cv)
    #     cv.fit(data_train)

    #     trf = cv.best_estimator_.named_steps["trf"]
    #     coefs_df = trf_to_dataframe(trf)
    #     coefs_df["dataset"] = dataset.name
    #     coefs_df.to_csv(f"coefs.{dataset.name.replace('/', '-')}.csv", index=False)

    #########

    # # TODO figure out shuffling. Can shuffle at the subject level ofc but not at the
    # # time series level.

    # # Nested cross-validation. Outer CV loop error on test set;
    # # inner CV loop estimates optimal hyperparameters.
    # outer_cv = KFold(n_splits=cfg.cv.n_outer_folds, shuffle=False)
    # fold_results = []
    # for i_split, (train_fold, test_fold) in enumerate(tqdm(outer_cv.split(data_train))):
    #     inner_cv = make_cv(model, cfg.cv)
    #     fold_results.append(inner_cv.fit(data_train[train_fold]))

    # if cfg.solver.type == "svd":
    #     model.fit(data_train)
    # else:
    #     model.partial_fit(data_train)


if __name__ == "__main__":
    main()