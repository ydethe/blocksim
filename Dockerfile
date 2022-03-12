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
# RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true
# RUN echo "tzdata tzdata/Areas select Europe" > preseed.txt && echo "tzdata tzdata/Zones/Europe select Paris" >> preseed.txt
# RUN debconf-set-selections preseed.txt && rm preseed.txt
# RUN apt-get update --allow-releaseinfo-change && apt-get install -yqq --no-install-recommends python3-dev python3-pip python3-venv libblas-dev liblapack-dev cmake gfortran gcc g++ make libproj-dev proj-data proj-bin libgeos-dev
# RUN python3 -m venv bs_env
# RUN source bs_env/bin/activate && python3 -m pip install -U pip && pip3 install scikit-build numpy==1.22.3 && pip3 install -r req1.txt -r req2.txt -r req3.txt && rm *.txt
