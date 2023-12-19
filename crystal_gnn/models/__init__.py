from crystal_gnn.models.cgcnn import CGCNN
from crystal_gnn.models.schnet import SCHNET
from crystal_gnn.models.megnet import MEGNET

_models = {
    "schnet": SCHNET,
    "cgcnn": CGCNN,
    "megnet": MEGNET,
}
