# pylint: disable-all
from sacred import Experiment

ex = Experiment("crystal-gnn")

alignn_config = {}


@ex.config
def config():
    exp_name = "crystal-gnn"
    seed = 0
    test_only = False

    # prepare_data # TODO: check unnecessary
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

    # dataloader
    batch_size = 64
    num_workers = 0
    pin_memory = True

    # model
    model_name = "cgcnn"  # "schnet", "cgcnn", "alignn"
    num_conv = 4
    hidden_dim = 128
    rbf_distance_dim = 50  # edge feature dimension
    batch_norm = True
    residual = True
    dropout = 0.0
    num_classes = 1  # if higher than 1, classification mode is activated

    # normalizer (only when num_classes == 1)
    mean = None  # when mean is None, it will be calculated from train data
    std = None  # when std is None, it will be calculated from train data

    # optimizer
    optimizer = "adamw"  # "adma", "sgd", "adamw"
    lr = 1e-3  # learning rate
    weight_decay = 1e-5
    scheduler = "reduce_on_plateau"  # "constant", "cosine", "reduce_on_plateau", "constant_with_warmup"

    # training
    devices = 1  # number of GPUs to use
    accelerator = "gpu"  # "cpu", "gpu"
    max_epochs = 150
    deterministic = True  # set True for reproducibility
    log_dir = "./crystal_gnn/logs"
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
def alignn():
    exp_name = "alignn"
    model_name = "alignn"


############
# matbench #
############
@ex.named_config
def matbench_schnet():
    exp_name = "schnet"
    model_name = "schnet"
    log_dir = "./crystal_gnn/logs/matbench"


@ex.named_config
def matbench_cgcnn():
    exp_name = "cgcnn"
    model_name = "cgcnn"
    log_dir = "./crystal_gnn/logs/matbench"


@ex.named_config
def matbench_alignn():
    exp_name = "alignn"
    model_name = "alignn"
    log_dir = "./crystal_gnn/logs/matbench"
