import deeplabcut as dlc
from pathlib import Path


project_dir = Path("/ceph/neuroinformatics/neuroinformatics/sirmpilatzen/DLC_HPC_test_data")
config_file_path = project_dir / "config.yaml"

DLC_PARENT_PATH = Path(dlc.auxiliaryfunctions.get_deeplabcut_path())
PRETRAINED_MODELS_DIR = DLC_PARENT_PATH / "pose_estimation_tensorflow" / "models" / "pretrained"
print(f"\nPretrained models directory: {PRETRAINED_MODELS_DIR}")

ALL_NET_TYPES = [
    "resnet_50",
    "resnet_101",
    "resnet_152",
    "mobilenet_v2_1.0",
    "mobilenet_v2_0.75",
    "mobilenet_v2_0.5",
    "mobilenet_v2_0.35",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
]


def model_is_downloaded(net_type: str):
    """Check if the specified model type exists in the 
    pretrained models directory
    
    Parameters
    ----------
    net_type : str
        The type of model to check for.
    """

    if net_type in ["resnet_50", "resnet_101", "resnet_152"]:
        net_type = net_type.replace("resnet", "resnet_v1")
    for file_or_folder in PRETRAINED_MODELS_DIR.iterdir():
        if net_type in file_or_folder.name:
            return True
    return False


for net_type in ALL_NET_TYPES:

    if model_is_downloaded(net_type):
        print(f"Model for {net_type} has already been downloaded.\n")
    else:
        dlc.create_training_dataset(config_file_path, net_type=net_type)
        if model_is_downloaded(net_type):
            print(f"Model for {net_type} was downloaded succesfully.\n")
        else:
            print(f"Failed to download model for {net_type}.\n")
