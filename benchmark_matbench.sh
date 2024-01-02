# https://github.com/materialsproject/matbench
project_name=crystal_gnn_v0.0.1
source=matbench
# 1. source="matbench" target="matbench_mp_gap"
python eval_matbench.py with cgcnn project_name=$project_name source=$source exp_name=cgcnn_matbench_mp_gap target=matbench_mp_gap
python eval_matbench.py with schnet project_name=$project_name source=$source exp_name=schnet_matbench_mp_gap target=matbench_mp_gap
python eval_matbench.py with megnet project_name=$project_name source=$source exp_name=megnet_matbench_mp_gap target=matbench_mp_gap

# 2. source="matbench" target="matbench_mp_e_form"
python eval_matbench.py with cgcnn project_name=$project_name source=$source exp_name=cgcnn_matbench_mp_e_form target=matbench_mp_e_form
python eval_matbench.py with schnet project_name=$project_name source=$source exp_name=schnet_matbench_mp_e_form target=matbench_mp_e_form
python eval_matbench.py with megnet project_name=$project_name source=$source exp_name=megnet_matbench_mp_e_form target=matbench_mp_e_form

# 3. source="matbench" target="matbench_log_gvrh"
python eval_matbench.py with cgcnn project_name=$project_name source=$source exp_name=cgcnn_matbench_log_gvrh target=matbench_log_gvrh
python eval_matbench.py with schnet project_name=$project_name source=$source exp_name=schnet_matbench_log_gvrh target=matbench_log_gvrh
python eval_matbench.py with megnet project_name=$project_name source=$source exp_name=megnet_matbench_log_gvrh target=matbench_log_gvrh

# 4. source="matbench" target="matbench_log_kvrh"
python eval_matbench.py with cgcnn project_name=$project_name source=$source exp_name=cgcnn_matbench_log_kvrh target=matbench_log_kvrh
python eval_matbench.py with schnet project_name=$project_name source=$source exp_name=schnet_matbench_log_kvrh target=matbench_log_kvrh
python eval_matbench.py with megnet project_name=$project_name source=$source exp_name=megnet_matbench_log_kvrh target=matbench_log_kvrh

# 5. source="matbench" target="matbench_perovskites"
python eval_matbench.py with cgcnn project_name=$project_name source=$source exp_name=cgcnn_matbench_perovskites target=matbench_perovskites
python eval_matbench.py with schnet project_name=$project_name source=$source exp_name=schnet_matbench_perovskites target=matbench_perovskites
python eval_matbench.py with megnet project_name=$project_name source=$source exp_name=megnet_matbench_perovskites target=matbench_perovskites


# 6. source="matbench" target="matbench_phonons"
python eval_matbench.py with cgcnn project_name=$project_name source=$source exp_name=cgcnn_matbench_phonons target=matbench_phonons
python eval_matbench.py with schnet project_name=$project_name source=$source exp_name=schnet_matbench_phonons target=matbench_phonons
python eval_matbench.py with megnet project_name=$project_name source=$source exp_name=megnet_matbench_phonons target=matbench_phonons

# 7. source="matbench" target="matbench_mp_is_metal"
python eval_matbench.py with cgcnn project_name=$project_name source=$source exp_name=cgcnn_matbench_mp_is_metal target=matbench_mp_is_metal
python eval_matbench.py with schnet project_name=$project_name source=$source exp_name=schnet_matbench_mp_is_metal target=matbench_mp_is_metal
python eval_matbench.py with megnet project_name=$project_name source=$source exp_name=megnet_matbench_mp_is_metal target=matbench_mp_is_metal