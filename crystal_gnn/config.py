# pylint: disable-all
from sacred import Experiment

ex = Experiment("crystal-gnn")

alignn_config = {}


@ex.config
def config():
    project_name = "crystal-gnn_test"  # for wandb
    exp_name = "crystal-gnn"
    seed = 0
    test_only = False

    # prepare_data
    source = "jarvis"  # "matbench"
    database_name = "dft_3d_2021"  # (Optional) for JARVIS
    target = "formation_energy_peratom"
    data_dir = "./data/"
    classification_threshold = None
    split_seed = 123
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    cutoff = 5.0
    max_neighbors = None  # None or int TODO:

    # dataloader
    batch_size = 64
    num_workers = 8
    pin_memory = True

    # model
    model_name = "cgcnn"  # "schnet", "cgcnn", "megnet"
    num_conv = 4
    hidden_dim = 128
    rbf_distance_dim = 80  # edge feature dimension
    batch_norm = True
    residual = True
    dropout = 0.0
    num_classes = 1  # if higher than 1, classification mode is activated

    # optimizer
    optimizer = "adamw"  # "adam", "sgd", "adamw"
    lr = 1e-3  # learning rate
    weight_decay = 1e-5
    scheduler = "reduce_on_plateau"  # "constant", "cosine", "reduce_on_plateau", "constant_with_warmup"

    # training
    num_nodes = 1  # number of nodes for distributed training
    devices = 1  # number of GPUs to use
    accelerator = "gpu"  # "cpu", "gpu"
    max_epochs = 100
    deterministic = True  # set True for reproducibility
    log_dir = "./logs"
    load_path = ""  # to load pretrained model
    resume_from = None  # resume from checkpoint


###########
# default #
###########
@ex.named_config
def schnet():
    exp_name = "schnet"
    model_name = "schnet"


@ex.named_config
def cgcnn():
    exp_name = "cgcnn"
    model_name = "cgcnn"


@ex.named_config
def megnet():
    exp_name = "megnet"
    model_name = "megnet"


@ex.named_config
def nequip():
    exp_name = "nequip"
    model_name = "nequip"


@ex.named_config
def matbench_log_gvrh_cgcnn():
    exp_name = "matbench_log_gvrh_cgcnn"
    source = "matbench"
    target = "matbench_log_gvrh"
    model_name = "cgcnn"
