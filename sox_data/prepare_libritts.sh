#!/usr/bin/env bash


log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


# General configuration
data_path="./data"                                          # Data Path for storage
#subsets="train-clean-100"                                   # Subsets for preparation
subsets="train-clean-100 train-clean-360 train-other-500"   # Subsets for preparation


# Create Datapath
log "mkdir -p ${data_path}"
mkdir -p ${data_path}


# Get files from URL
for subset in ${subsets}
do
    log "wget https://www.openslr.org/resources/60/${subset}.tar.gz -P ${data_path}"
    wget https://www.openslr.org/resources/60/${subset}.tar.gz -P ${data_path}
done


# Unzip files
for subset in ${subsets}
do
    log "tar -zxf ${data_path}/${subset}.tar.gz -C ${data_path}"
    tar -zxf ${data_path}/${subset}.tar.gz -C ${data_path}
done


# Remove zip files
for subset in ${subsets}
do
    log "rm -rf ${data_path}/${subset}.tar.gz"
    rm -rf ${data_path}/${subset}.tar.gz
done


# Modify directory name
log "mv ${data_path}/LibriTTS ${data_path}/libritts"
mv ${data_path}/LibriTTS ${data_path}/libritts
