FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    wget \
    ca-certificates \
    build-essential \
    git \
    libssl-dev \
    pandoc \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic \
    libffi-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 && \
    ln -s /usr/bin/python3.12 /usr/bin/python && \
    ln -s /usr/local/bin/pip /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get purge -y python3-blinker || true

COPY requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12-dev

# Install JAX with CUDA 12 support
#RUN pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#Jupyter to get more stupider...
RUN pip install --no-cache-dir "markupsafe<2.1" \
    && pip install --upgrade jinja2 jupyterlab jupyter_server ipywidgets jupytext ipykernel nbconvert jupyterlab_widgets ipympl nbcommands nbconvert

WORKDIR /workspace
#CMD ["python"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
