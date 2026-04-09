#!/bin/bash
base='/home/hojaeson_umass_edu/project/vllm-test/ref/lotus-experiment/pipelines/lotus'
log_dir='/home/hojaeson_umass_edu/project/vllm-test/ref/lotus-experiment/pipelines/lotus/script/logs'

mkdir -p "$log_dir"
python3 -u "${base}/enron_filter_filter_map.py" 2>&1 | tee "${log_dir}/enron_filter_filter_map.log"
python3 -u "${base}/enron_filter_filter_map_2.py" 2>&1 | tee "${log_dir}/enron_filter_filter_map_2.log"
