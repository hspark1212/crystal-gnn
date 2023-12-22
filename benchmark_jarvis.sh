# https://pages.nist.gov/jarvis/databases
# 1. source="jarvis" database_name="dft_3d_2021" target="formation_energy_peratom"
python run.py with source=jarvis model_name=schnet exp_name=schnet_formation_energy_peratom database_name=dft_3d_2021 target=formation_energy_peratom
python run.py with source=jarvis model_name=cgcnn exp_name=cgcnn_formation_energy_peratom database_name=dft_3d_2021 target=formation_energy_peratom
python run.py with source=jarvis model_name=megnet exp_name=megnet_formation_energy_peratom database_name=dft_3d_2021 target=formation_energy_peratom


# 2. source="jarvis" database_name="dft_3d_2021" target="optb88vdw_bandgap"
python run.py with source=jarvis model_name=schnet exp_name=schnet_optb88vdw_bandgap database_name=dft_3d_2021 target=optb88vdw_bandgap
python run.py with source=jarvis model_name=cgcnn exp_name=cgcnn_optb88vdw_bandgap database_name=dft_3d_2021 target=optb88vdw_bandgap
python run.py with source=jarvis model_name=megnet exp_name=megnet_optb88vdw_bandgap database_name=dft_3d_2021 target=optb88vdw_bandgap

# 3. source="jarvis" database_name="dft_3d_2021" target="ehull"
python run.py with source=jarvis model_name=schnet exp_name=schnet_ehull database_name=dft_3d_2021 target=ehull
python run.py with source=jarvis model_name=cgcnn exp_name=cgcnn_ehull database_name=dft_3d_2021 target=ehull
python run.py with source=jarvis model_name=megnet exp_name=megnet_ehull database_name=dft_3d_2021 target=ehull

# 4. source="jarvis" database_name="dft_3d_2021" target="bulk_modulus_kv" 
python run.py with source=jarvis model_name=schnet exp_name=schnet_bulk_modulus_kv database_name=dft_3d_2021 target=bulk_modulus_kv
python run.py with source=jarvis model_name=cgcnn exp_name=cgcnn_bulk_modulus_kv database_name=dft_3d_2021 target=bulk_modulus_kv
python run.py with source=jarvis model_name=megnet exp_name=megnet_bulk_modulus_kv database_name=dft_3d_2021 target=bulk_modulus_kv

# 5. source="jarvis" database_name="dft_3d_2021" target="shear_modulus_gv"
python run.py with source=jarvis model_name=schnet exp_name=schnet_shear_modulus_gv database_name=dft_3d_2021 target=shear_modulus_gv
python run.py with source=jarvis model_name=cgcnn exp_name=cgcnn_shear_modulus_gv database_name=dft_3d_2021 target=shear_modulus_gv
python run.py with source=jarvis model_name=megnet exp_name=megnet_shear_modulus_gv database_name=dft_3d_2021 target=shear_modulus_gv