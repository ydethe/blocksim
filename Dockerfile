# docker build -t blocksim .
# docker tag blocksim:latest ydethe/blocksim:latest
# docker push ydethe/blocksim:latest
# docker system prune --volumes
# FROM ubuntu:focal
FROM continuumio/miniconda3
WORKDIR /app
SHELL ["/bin/bash", "-c"]
ADD environment_test.yml /app/environment_test.yml
RUN conda install -y mamba -n base -c conda-forge
RUN mamba env create -f environment_test.yml

