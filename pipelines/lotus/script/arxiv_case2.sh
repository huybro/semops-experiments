base='..'
mkdir -p logs
# python3 -u ${base}/arxiv_case_2_filter.py 2>&1 | tee logs/arxiv_case_2_filter.log
python3 -u ${base}/arxiv_case_2_filter_join.py 2>&1 | tee logs/arxiv_case_2_filter_join.log
python3 -u ${base}/arxiv_case_2_filter_join_map.py 2>&1 | tee logs/arxiv_case_2_filter_join_map.log
