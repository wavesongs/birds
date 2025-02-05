FROM python:3.10

RUN apt-get update
RUN apt-get install -y python3-pip

RUN git clone https://github.com/wavesongs/wavesongs

# add requirements.txt, written this way to gracefully ignore a missing file
COPY requirements.txt .
RUN ([ -f requirements.txt ] \
    && pip3 install --no-cache-dir -r requirements.txt) \
    || pip3 install --no-cache-dir jupyter jupyterlab 

USER root

# Unpack and install the kernel
RUN cd wavesongs \
    && pip install -e .

# Set up the user environment
ENV NB_USER=saguileran
ENV NB_UID=1000
ENV HOME=/home/$NB_USER

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid $NB_UID \
    $NB_USER

COPY . $HOME
RUN chown -R $NB_UID $HOME

USER $NB_USER

# Launch the notebook server
WORKDIR $HOME
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser"]