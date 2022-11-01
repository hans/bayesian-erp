# Extended from https://github.com/huggingface/transformers/blob/main/docker/transformers-pytorch-cpu/Dockerfile
# because I needed a higher Python version

FROM ubuntu:22.04

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir \
        jupyter jupytext torchtyping typeguard icecream \
        pytest pytest-cov coverage \
        transformers \
        hydra-core hydra-optuna-sweeper optuna tensorboard \
        pandas scipy seaborn \
        mne scikit-learn \
        textgrid

CMD ["/bin/bash"]
