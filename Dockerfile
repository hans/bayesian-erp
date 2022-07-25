FROM jrgauthier/pyro:cpu-3.9

RUN pip install pandas scipy jupyterlab jupytext torchtyping typeguard icecream transformers[torch]
RUN pip install seaborn mypy pytest mne scikit-learn
