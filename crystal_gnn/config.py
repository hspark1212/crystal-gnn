# pylint: disable-all
from sacred import Experiment

ex = Experiment("crystal-gnn")

alignn_config = {}


@ex.config
def config():
    exp_name = "crystal-gnn"
    seed = 0
    test_only = False

    # prepare_data
    source = "jarvis"
    database_name = "dft_3d_2021"
    target = "formation_energy_peratom"
    data_dir = "./crystal_gnn/data"
    classification_threshold = None
    split_seed = 123
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    keep_data_order = False

    # dataset
    compute_line_graph = True
    neighbor_strategy = "k-nearest"
    cutoff = 8.0
    max_neighbors = 12
    use_canonize = True

    # dataloader
    batch_size = 64
    num_workers = 0  # This should be 0 to use dataloader with dgl graph
    pin_memory = True
    use_ddp = True

    # model
    model_name = "cgcnn"  # "schnet", "cgcnn", "alignn"
    num_conv = 4
    hidden_dim = 256
    rbf_distance_dim = 80  # RDF expansion dimension for edge distance
    rbf_triplet_dim = 40  # RDF expansion dimension for triplet angle
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
