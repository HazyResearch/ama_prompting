
for task in WSC_variants; do
    python3 ${task}.py \
	--run_zeroshot 0 \
	--run_fewshot 0 \
	--run_decomp 1 \
	--num_boost 10 \
	--k_shot 3 \
	--output_metrics_file ../ama_logs/metrics.json \
	--save_dir ../ama_logs/ama_final_runs \
	--overwrite_data 1 \
	--overwrite_boost_exs 1 \
	--num_run -1 \
	--client_connection http://127.0.0.1:5000
done;