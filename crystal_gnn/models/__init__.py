from crystal_gnn.models.cgcnn import CGCNN
from crystal_gnn.models.schnet import SCHNET
from crystal_gnn.models.megnet import MEGNET
from crystal_gnn.experimental.alignn import ALIGNN

_models = {
    "schnet": SCHNET,
    "cgcnn": CGCNN,
    "megnet": MEGNET,
    "alignn": ALIGNN,
}
