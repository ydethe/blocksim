FROM continuumio/miniconda3:4.9.2
ADD . /app/
WORKDIR /app
SHELL ["/bin/bash", "-c"]
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true
RUN echo "tzdata tzdata/Areas select Europe" > preseed.txt
RUN echo "tzdata tzdata/Zones/Europe select Berlin" >> preseed.txt
RUN debconf-set-selections preseed.txt
RUN apt-get update --allow-releaseinfo-change
RUN apt-get install -yqq --no-install-recommends curl
RUN conda install -y mamba -n base -c conda-forge
RUN mamba env create -f environment_test.yml
