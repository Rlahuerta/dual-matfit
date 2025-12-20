ARG MAIN_REPO=/opt/repo
ARG PYSETUP_PATH=/opt/pysetup

FROM python:3.12-bookworm AS linux

LABEL maintainer="RILAH"

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    LANGUAGE='en_US:en' \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q && \
    apt-get upgrade -y --fix-missing && \
    apt-get install -y --no-install-recommends \
    wget curl ca-certificates git xvfb python3-dev python3-llvmlite \
    texlive texlive-latex-extra texlive-fonts-recommended dvipng \
    python3-pip python3-venv python3-tk \
    python3-cachecontrol python3-debian python-is-python3 && \
    update-ca-certificates dos2unix

# Install NALA
RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends --fix-missing -o Acquire::http::Pipeline-Depth=0 nala && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN nala update && nala upgrade --assume-yes
 
# Install of basic and need packages
RUN nala install --update --assume-yes bzip2 libglib2.0-0 \
    libsecret-1-dev pkg-config openssh-client subversion vim \
    procps nano locales locales-all apt-utils doxygen \
    build-essential gcc g++ gfortran software-properties-common \
    autoconf automake libre2-dev libssl-dev libtool libcurl4-openssl-dev \
    rapidjson-dev patchelf zlib1g-dev

# Install HDF(4, 5) libraries
RUN nala install --update --assume-yes hdf4-tools libhdf5-dev hdf5-tools

# Install Math Libraries
RUN nala install --update --assume-yes \
    libffi-dev libbz2-dev libsqlite3-dev libblas-dev liblapacke-dev liblapack-dev libmetis-dev \
    libeigen3-dev libboost-dev libb64-dev 

# Extras Libraries for IpOpt
# https://installati.one/ubuntu/22.04/coinor-libipopt1v5/
RUN nala update && nala install --update --assume-yes \
    libblas-dev libatlas-base-dev liblapack-dev \
    libmetis-dev libibnetdisc-dev libboost-dev libgeos-dev \
    libmumps-dev mumps-test coinor-libipopt-dev coinor-libipopt1v5 

# Latex libraries
RUN nala install --update --assume-yes texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# Get basic python package distribution
RUN apt-get autoremove && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

FROM linux AS main

ARG MAIN_REPO
ARG PYSETUP_PATH

USER root

ENV PYSETUP_PATH=${PYSETUP_PATH}
ENV PATH="${PATH}:${PYSETUP_PATH}/.venv/bin"
ENV PENV_ACT="${PYSETUP_PATH}/.venv/bin/activate"

USER root

# Generate locale
RUN locale-gen en_US.UTF-8

ADD  . $PYSETUP_PATH
WORKDIR $PYSETUP_PATH

RUN pip install -U pip poetry \
    && poetry config virtualenvs.path ${PENV_ACT} \
    && poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true --local \
    && poetry install

# Activate environment at session start
RUN echo "source ${PYSETUP_PATH}/.venv/bin/activate" >> ~/.bashrc && . $PENV_ACT

# Layer for vestas topfarm test
FROM main AS build

ARG PYSETUP_PATH
ENV PENV_ACT=${PYSETUP_PATH}/.venv/bin/activate

# Get basic python package distribution
RUN apt-get autoremove && apt-get clean \
    && rm -rf $PYSETUP_PATH/Results/* /tmp/* /var/tmp/*

# Activate environment at session start
RUN echo "source ${PYSETUP_PATH}/.venv/bin/activate" >> ~/.bashrc

WORKDIR $PYSETUP_PATH

# EXPOSE 8888
SHELL ["/bin/bash", "-c"]
