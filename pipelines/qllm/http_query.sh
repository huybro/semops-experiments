#!/bin/bash

# start=$(date +%s.%N)
# curl -X POST http://localhost:8003/v1/semantic/query_ref \
#   -H "Content-Type: application/json" \
#   -d '{
#     "query": "df.sem_filter().sem_map()",
#     "data_path": "/scratch/hojaeson_umass/kagglehub/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
#   }'

  
# end=$(date +%s.%N)

# elapsed=$(echo "$end - $start" | bc)
# echo "query_ref Elapsed time: $elapsed seconds"


#"data_path": "/scratch/hojaeson_umass/kagglehub/snehaanbhawal/resume-dataset/versions/1/Resume/Resume.csv"
# /scratch/hojaeson_umass/kagglehub/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/versions/1/imdb_sample_500_texts

start=$(date +%s.%N)
curl -X POST http://localhost:8003/v1/semantic/query \
  -H "Content-Type: application/json" \
  -d '{
    "ops": "df.sem_filter().sem_map()",
    "data_path": "/scratch/hojaeson_umass/kagglehub/spsayakpaul/arxiv-paper-abstracts/versions/2/arxiv_txt"
  }'

  
end=$(date +%s.%N)

elapsed=$(echo "$end - $start" | bc)
echo "queryElapsed time: $elapsed seconds"