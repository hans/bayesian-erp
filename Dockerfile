FROM jrgauthier/pyro:cpu-3.9

RUN pip install pandas scipy jupyterlab jupytext
RUN pip install torchtyping typeguard icecream
