base='..'
mkdir -p logs
python3 -u ${base}/job_matching_case_1_filter.py  2>&1 | tee logs/job_matching_case_1_filter.log
python3 -u ${base}/job_matching_case_1_filter_join.py  2>&1 | tee logs/job_matching_case_1_filter_join.log
python3 -u ${base}/job_matching_case_1_filter_join_map.py  2>&1 | tee logs/job_matching_case_1_filter_join_map.log
