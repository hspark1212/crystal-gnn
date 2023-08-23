# 1. evaluate models on all matbench datasets
python eval_matbench.py with model_name=schnet target=all log_dir="crystal_gnn/logs_matbench"
python eval_matbench.py with model_name=cgcnn target=all log_dir="crystal_gnn/logs_matbench"
python eval_matbench.py with model_name=alignn target=all log_dir="crystal_gnn/logs_matbench"

# 2. evaluate models with specific target
# python eval_matbench.py with model_name=schnet target=matbench_mp_gap
# python eval_matbench.py with model_name=cgcnn target=matbench_mp_e_form
# python eval_matbench.py with model_name=alignn target=matbench_phonons
