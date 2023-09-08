
<div align="center">

<h1> 🌐 Crystal Graph Neural Networks 💎 </h1>

A banchmarking framework for Crystal Graph Neural Networks (CGNN)

built on top of `PyTorch Lightning` and `Deep Graph Library (DGL)`

</div>

## 📜 Table of Contents
- [1. Introduction](#1-introduction)
- [2. Installation](#2-installation)
- [3. Usage](#3-usage)
    - [1. Benchmarking with the Jarvis-Tools Database](#1-benchmarking-with-the-jarvis-tools-database)
    - [2. Benchmarking with the Matbench Database](#2-benchmarking-with-the-matbench-database)
- [4. Results and Benchmarks](#4-results-and-benchmarks)
    - [1. Jarvis-Tools Database](#1-jarvis-tools-database)
    - [2. Matbench Database](#2-matbench-database)
- [5. Contributing](#5-contributing)
    - [Adding a New Model](#🤗-adding-a-new-model)
- [6. License](#6-license)
---

<br>

## 1. Introduction

Welcome to the world of **Crystal Graph Neural Networks (CGNN)**! This exciting field of research finds its applications in diverse areas, ranging from material science to chemistry. This repository has been created to serve as a comprehensive benchmarking framework tailored for CGNNs. It aims to facilitate the evaluation of CGNN performance, making it easier to assess their capabilities. 

This repository have integrated well-known models and databases, such as [Jarvis-Tools](https://pages.nist.gov/jarvis/databases/) and [Matbench](https://matbench.materialsproject.org/), making use of the power and flexibility of `PyTorch Lightning` and `DGL`.



### 🚀 Features
- **Supported Models**: (more models will be added)
  - [SCHNET](https://arxiv.org/abs/1706.08566)
  - [CGCNN](https://arxiv.org/abs/1710.10324)
  - [ALIGNN](https://www.nature.com/articles/s41524-021-00650-1)
  
- **database Integrated**:
  - [Jarvis-Tools database](https://pages.nist.gov/jarvis/databases/)
  - [Matbench](https://matbench.materialsproject.org/)

- **Performance Evaluation**: Post-benchmarking, model performances can be visualized and compared using Tensorboard.

---

<br>

## 2. Installation

### **Method 1**: Reproducible Installation

This method ensures a more reproducible setup by using a pre-defined environment.

```bash
git clone https://github.com/hspark1212/crystal-gnn.git
cd crystal-gnn
conda env create -f environment.yml
```

<br>

### **Method 2**: Standard Installation

This method is more flexible without strict dependencies.

Step 1: Make Conda Environment with Python 3.9

```bash
conda create -n crystal-gnn python=3.9
conda activate crystal-gnn
```

Step 2: Clone the Repository & Install Dependencies
```bash
git clone https://github.com/hspark1212/crystal-gnn.git
cd crystal-gnn
pip install -r requirements.txt
```
---

<br>

## 3. Usage

### **1) Benchmarking with the Jarvis-Tools Database**

The script [benchmark_jarvis.sh](benchmark_jarvis.sh) is an example of how to run the benchmarking across all models on the Jarvis-Tools database. Once the training is completed, results are stored in the `./crystal-gnn/logs/` directory. Visualization of these results is made possible through `Tensorboard`.

<br>

#### **Example 1:** To benchmark the CGCNN model on the `formation_energy_per_atom` task, run the following command:

```bash
python run.py with cgcnn exp_name=cgcnn_formation_energy_peratom database_name=dft_3d_2021 target=formation_energy_peratom
```

This command will download the Jarvis-Tools database in `./crystal-gnn/data` and train the CGCNN model on the `formation_energy_per_atom` task. The trained model will be saved in `./logs/cgcnn_formation_energy_peratom/`. 

<br>

#### **Example 2:** To benchmark the ALIGNN model on the `band gap` task and set `hidden_dim` to `32`, run the following command:

```bash
python run.py with alignn exp_name=alignn_optb88vdw_bandgap database_name=dft_3d_2021 target=optb88vdw_bandgap hidden_dim=32
```

Just as you can adjust the hidden_dim parameter, various parameters related to targets, training, logs, and individual models can be modified in the [configs.py](crystal_gnn/config.py) file.

<br>

#### **Viewing Results with Tensorboard:**

After completing the benchmarks, launch Tensorboard with the following command to visualize the results:

```bash
tensorboard --logdir=./crystal-gnn/logs/ --bind_all
```

By entering the provided URL into your web browser (usually something like `http://localhost:6006/`), you can access Tensorboard's graphical interface and review the benchmark outcomes.

<br>

### **2) Benchmarking with the Matbench Database**

The script [eval_matbench.py](eval_matbench.py) is to benchmark using the Matbench database, which includes 13 different tasks. Among these tasks, only 7 tasks involve utilizing structural information as inputs. Consequently, this script is designed to perform benchmarking on 6 regression tasks and 1 classification task, listed as follows:

    matbench_log_gvrh
    matbench_log_kvrh
    matbench_mp_e_form
    matbench_mp_gap
    matbench_mp_is_metal
    matbench_perovskites
    matbench_phonons

<br>

#### **Example 1**: To run benchmarks for the CGCNN model on the listed 7 tasks, run the following command:

```bash
python eval_matbench.py with model_name=cgcnn target=all
```

<br>

#### **Example 2**: To run benchmarks for the CGCNN model on the specific task `matbench_mp_gap`, run the following command:

```bash
python eval_matbench.py with model_name=cgcnn target=matbench_mp_gap
```

After training, the model along with the corresponding log files will be stored at `./logs/{model_name}_matbench_{task_name}`. Furthermore, the JSON files, intended for submission to the Matbench leaderboard, will be stored at `./logs/{model_name}_{task_name}/results_{model_name}_{task_name}.json.gz`.

---

<br>

## 4. Results and Benchmarks

The benchmarks are implemented with the following dependencies:
    
  - **OS**: Ubuntu 20.04
  - **GPU**: NVIDIA GeForce RTX 2080 Ti
  - **CUDA**: 11.7
  - **cuDNN**: 8.9.3
  - **python**: 3.9
  - **Dependencies**:
    ``` 
    dgl==1.1.2 (cu117)
    torch==2.0.1
    pytorch-lightning==2.0.6
    ```

For reproducibility, our benchmarks have been verified using `pytorch-lightning` configurations (`seed_everything=0` and `deterministic=True`). It will guarantee the same results in the same machine and environment.

<br>

### **1) Jarvis-Tools Database**

| Task                        | SCHNET       | CGCNN       | ALIGNN      |
|-----------------------------|--------------|-------------|-------------|
| formation_energy_per_atom   | 0.057 (0.98) | 0.039 (0.99)| 0.034 (0.99)|
| Band gap                    | 0.194 (0.85) | 0.147 (0.89)| 0.133 (0.89)|
| energy above hull           | 0.162 (0.94) | 0.106 (0.98)| 0.066 (0.99)|
| bulk modulus                | 15.159 (0.83)| 13.271 (0.85)| 14.043 (0.85)|
| shear modulus               | 11.710 (0.70)| 10.960 (0.74)| 10.497 (0.73)|

This results are stored in [`results/benchmark_jarvis`](results/benchmark_jarvis/), which can be visualized with the following command `tensorboard --logdir=./crystal-gnn/results/benchmark_jarvis --bind_all`.

<br>

### **2) Matbench Database** 

| Task                  | SCHNET       | CGCNN       | ALIGNN  (will be updated soon)     |
|-----------------------|--------------|-------------|-------------|
| matbench_log_gvrh     | 0.097 (0.84)| 0.090 (0.86)| |
| matbench_log_kvrh     | 0.075 (0.87)| 0.068 (0.88)| |
| matbench_mp_e_form    | 0.039 (-56.24)| 0.027 (-37.33)| |
| matbench_mp_gap       | 0.287 (0.57)| 0.206 (0.86)| |
| matbench_perovskites  | 0.051 (0.98)| 0.039 (0.99)| |
| matbench_phonons      | 87.952 (0.81)| 65.544 (0.91)| |
| matbench_mp_is_metal  | [0.898]| [0.899]| |
* The results are computed using the average of five folds.

> Note: Results are displayed as MAE (R2) for regression tasks and [Accuracy] for classification tasks.

---

<br>

## 5. Contributing

Other databases and models are welcome to be added to the benchmarking. If you would like to contribute, please open an issue or submit a pull request.

### 🤗 **Adding a New Model** 

The following steps are required to add a new model to the benchmarking: 

<br>

**Step 1**: Create a new model class in the [models](crystal_gnn/models/) directory that inherits from the [BaseModule](crystal_gnn/models/base_module.py) class.
The `BaseModule` class is a wrapper class that inherits from `pytorch-lightning`'s `LightningModule` class and provides a convenient interface for training and evaluation. The `BaseModule` class also provides a `forward` method that takes in a `dgl.DGLGraph` object and returns a prediction. The `forward` method is used for inference and evaluation. 

```python
from crystal_gnn.models.base_module import BaseModule

class MyModel(BaseModule):
    def __init__(self, _config: Dict[str, Any]):
        super().__init__(_config)
        # Initialize your model here

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Perform inference and return a prediction
        graph = batch['graph']
        line_graph = batch['line_graph'] # optional
        # Message passing ...
```

The batch includes graph and line graph representations of the input data. The graph is a `dgl.DGLGraph` object that contains the following node and edge features:

- `g.ndata['atomic_number']`: Atomic number of each node (torch.Tensor, shape = (N, 1))
- `g.ndata['coord']`: Atomic coordinates of each node (torch.Tensor, shape = (N, 3))
- `g.edata['distance']`: Distance between two nodes (torch.Tensor, shape = (E, 1))
- `g.edata['angle']`: Angle between three nodes (torch.Tensor, shape = (E, 1)) (If `line_graph` is provided)

For simplicity, the hyperparmeters of models are defined as follows:

- `model_name`: Name of the model (str, "schnet", "cgcnn", "alignn")
- `num_conv`: Number of convolution layers (int, default = 4)
- `hidden_dim`: Hidden dimension of the model (int, default = 256)
- `rbf_distance_dim`: RDF expansion dimension for edge distance (int, default = 80)
- `rbf_triplet_dim`: RDF expansion dimension for triplet angle (int, default = 40)
- `batch_norm`: Whether to use batch normalization (bool, default = True)
- `residual`: Whether to use residual connections (bool, default = True)
- `dropout`: Dropout rate (float, default = 0.0)

 The hyperparmeters can be added and modified in the [configs.py](crystal_gnn/config.py) file.

<br>

**Step 2**: Add the model to the `_model` dictionary in the [crystal_gnn/models/__init__.py](crystal_gnn/models/__init__.py) file.

```python
from crystal_gnn.models.cgcnn import CGCNN
from crystal_gnn.models.alignn import ALIGNN
from crystal_gnn.models.schnet import SCHNET
from crystal_gnn.models.my_model import MyModel

_models = {
    "cgcnn": CGCNN,
    "alignn": ALIGNN,
    "schnet": SCHNET,
    "my_model": MyModel
}
```

<br>

**Step 3**:  run the benchmarking script with the new model. For example, to run the benchmarking with the `my_model` model, run the following command:

```bash
python run.py with model_name=my_model exp_name=my_model_formation_energy_peratom database_name=dft_3d_2021 target=formation_energy_peratom
```

---

<br>

## 6. License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

