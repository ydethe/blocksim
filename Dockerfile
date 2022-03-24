# docker build -t blocksim .
# docker tag blocksim:latest ydethe/blocksim:latest
# docker push ydethe/blocksim:latest
# docker system prune --volumes
FROM ubuntu:focal
WORKDIR /
SHELL ["/bin/bash", "-c"]
RUN export DEBIAN_FRONTEND=noninteractive DEBCONF_NONINTERACTIVE_SEEN=true
RUN echo "tzdata tzdata/Areas select Europe" > preseed.txt
RUN echo "tzdata tzdata/Zones/Europe select Paris" >> preseed.txt
RUN debconf-set-selections preseed.txt
RUN apt-get update --allow-releaseinfo-change && apt-get install -yqq --no-install-recommends pipx python3.8-dev python3.8-venv libblas-dev liblapack-dev cmake gfortran gcc g++ make libproj-dev proj-data proj-bin libgeos-dev curl
RUN pipx install poetry
