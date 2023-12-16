from crystal_gnn.datamodules.jarvis_datamodule import JarvisDataModule
from crystal_gnn.datamodules.matbench_datamodule import MatbenchDataModule

_datamodules = {
    "jarvis": JarvisDataModule,
    "matbench": MatbenchDataModule,
}
