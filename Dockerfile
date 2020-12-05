
FROM stablebaselines/stable-baselines3-cpu:0.10.0

RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    libmysqlclient-dev \
    xvfb \
    libxrandr2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY requirements.txt /tmp/


RUN \
    mkdir -p ${CODE_DIR}/rl_zoo && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install git+https://github.com/eleurent/highway-env && \
    rm -rf $HOME/.cache/pip

ENV PATH=$VENV/bin:$PATH

COPY . /tmp/
# RUN chmod +x /tmp/entrypoint.sh
# ENTRYPOINT ["/tmp/entrypoint.sh"]
# ENTRYPOINT python /tmp/lunarlander/lunarlander.py && /bin/bash
# ENTRYPOINT python /tmp/mountaincar/hptuning.py && /bin/bash
ENTRYPOINT python /tmp/BipedalWalker-v3/tuneSAC.py && /bin/bash

