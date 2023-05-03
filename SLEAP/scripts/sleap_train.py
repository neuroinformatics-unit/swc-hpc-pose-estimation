import sleap
from pathlib import Path
from typing import Optional

from sleap.nn.config import (
    TrainingJobConfig,
    UNetConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
)
from sleap.nn.training import Trainer


def split_labels(labels_path: str, train_fraction: float = 0.8) -> dict:
    """Load labels from SLEAP project file, split into train/val/test sets,
    and save as separate SLEAP project files. The split files will be saved
    in the same directory as the original file, with the split name appended
    to the file name (e.g., "my_labels.slp" -> "my_labels.train.pkg.slp")

    Parameters
    ----------
    labels_path : str
        Path to SLEAP project file.
    train_fraction : float, optional
        Fraction of labels to use for training set, by default 0.8
        Vaidation and test sets will each contain half of the remaining labels.

    Returns
    -------
    dict
        Dictionary with keys "train", "val", and "test" containing
        paths to the saved SLEAP project files.
    """
    labels = sleap.load_file(labels_path)
    labels_train, labels_val_test = labels.split(n=train_fraction)
    labels_val, labels_test = labels_val_test.split(n=0.5)
    labels_split_data = {
        "train": labels_train,
        "val": labels_val,
        "test": labels_test,
    }
    labels_split_paths = {}
    for split, labels in labels_split_data.items():
        labels_split_path = labels_path.replace(".slp", f".{split}.pkg.slp")
        labels.save(labels_split_path, with_images=True)
        labels_split_paths[split] = labels_split_path
    return labels_split_paths


def update_config_labels(
    cfg: TrainingJobConfig, labels_split_paths: dict
) -> TrainingJobConfig:
    """Update training job configuration with paths to split labels.

    Parameters
    ----------
    cfg : sleap.nn.config.training_job.TrainingJobConfig
        Training job configuration.
    labels_split_paths : dict
        Dictionary with keys "train", "val", and "test" containing
        paths to the saved SLEAP project files.

    Returns
    -------
    sleap.nn.config.training_job.TrainingJobConfig
        Updated training job configuration.
    """
    # Add paths to split labels
    cfg.data.labels.training_labels = labels_split_paths["train"]
    cfg.data.labels.validation_labels = labels_split_paths["val"]
    cfg.data.labels.test_labels = labels_split_paths["test"]
    return cfg


def update_config_outputs(
    cfg: TrainingJobConfig, run_name_prefix: str, models_dir: str = "models"
) -> TrainingJobConfig:
    """Update training job configuration with model outputs options.

    Parameters
    ----------
    cfg : sleap.nn.config.training_job.TrainingJobConfig
        Training job configuration.
    run_name_prefix : str
        Prefix to prepend to run name.
    """
    # configure outputs
    cfg.outputs.run_name_prefix = run_name_prefix
    cfg.outputs.runs_folder = models_dir
    cfg.outputs.save_outputs = True
    cfg.outputs.save_visualizations = True
    cfg.outputs.checkpointing.initial_model = True
    cfg.outputs.checkpointing.best_model = True
    return cfg


def update_config_rotation(
    cfg: TrainingJobConfig,
    rotation_min_angle: float = -180,
    rotation_max_angle: float = 180,
) -> TrainingJobConfig:
    """Update training job configuration with rotation augmentation options.
    The -180 to 180 degree range is suitable for top-down camera views.

    Parameters
    ----------
    cfg : sleap.nn.config.training_job.TrainingJobConfig
        Training job configuration.
    rotation_min_angle : float, optional
        Minimum rotation angle in degrees, by default -180
    rotation_max_angle : float, optional
        Maximum rotation angle in degrees, by default 180
    """
    cfg.optimization.augmentation_config.rotate = True
    cfg.optimization.augmentation_config.rotation_min_angle = (
        rotation_min_angle
    )
    cfg.optimization.augmentation_config.rotation_max_angle = (
        rotation_max_angle
    )
    return cfg


def configure_model(
    model_type: str,
    labels_split_paths: dict,
    skeleton_path: Optional[str] = None,
    anchor_part: str = "centre",
    input_scaling: float = 1.0,
    n_epochs: int = 50,
    batch_size: int = 4,
    run_name_prefix: Optional[str] = None,
    models_dir: str = "models",
) -> TrainingJobConfig:
    """Configure training job for a SLEAP model.

    Parameters
    ----------
    model_type : str
        Type of model to train.
        Must be one of "centroid" or "centered_instance".
    labels_split_paths : dict
        Dictionary with keys "train", "val", and "test" containing
        paths to the saved SLEAP project files.
    skeleton_path : str, optional
        Path to skeleton file, by default None
        It must be provided if model_type is "centered_instance".
    anchor_part : str, optional
        Name of body part to use for centering instances, by default "centre"
    input_scaling : float, optional
        Scaling factor for input images, by default 1.0
    n_epochs : int, optional
        Number of training epochs, by default 50
    batch_size : int, optional
        Batch size, by default 4
    run_name_prefix : str, optional
        Prefix to prepend to run name, by default None
    models_dir : str, optional
        Directory to save models, by default "models"

    Returns
    -------
    sleap.nn.config.training_job.TrainingJobConfig
        Training job configuration object.
    """

    if model_type not in ["centroid", "centered_instance"]:
        raise ValueError(
            f"Model type {model_type} not recognized."
            "Must be one of 'centroid' or 'centered_instance'."
        )

    # Initalise default training config
    cfg = TrainingJobConfig()

    # Add paths to split labels
    cfg = update_config_labels(cfg, labels_split_paths)
    if model_type == "centered_instance":
        skeleton = sleap.Skeleton.load_json(skeleton_path.as_posix())
        cfg.data.labels.skeletons = [skeleton]

    # Data preprocessing, augmentation, and optimisation parameters
    cfg.data.preprocessing.input_scaling = (input_scaling,)
    cfg.data.preprocessing.ensure_grayscale = True
    cfg = update_config_rotation(cfg)
    cfg.data.instance_cropping.center_on_part = anchor_part
    cfg.data.instance_cropping.crop_size_detection_padding = 32
    cfg.optimization.epochs = n_epochs
    cfg.optimization.batch_size = batch_size

    # configure NN backbone (UNet) and head
    if model_type == "centroid":
        cfg.model.backbone.unet = UNetConfig(
            max_stride=16,
            filters=16,
            filters_rate=2.00,
            output_stride=2,
            up_interpolate=True,
        )
        cfg.model.heads.centroid = CentroidsHeadConfig(
            anchor_part=anchor_part, sigma=2.5, output_stride=2
        )
    elif model_type == "centered_instance":
        cfg.model.backbone.unet = UNetConfig(
            max_stride=16,
            filters=24,
            filters_rate=2.00,
            output_stride=4,
            up_interpolate=True,
        )
        cfg.model.heads.centered_instance = CenteredInstanceConfmapsHeadConfig(
            anchor_part=anchor_part,
            sigma=2.5,
            output_stride=4,
            loss_weight=1.0,
        )

    # configure model outputs
    cfg = update_config_outputs(cfg, run_name_prefix, models_dir)

    return cfg


def train_model(train_config: TrainingJobConfig) -> None:
    """Train SLEAP model using the given training configuration.

    Parameters
    ----------
    train_config : sleap.nn.config.training_job.TrainingJobConfig
        Training job configuration for SLEAP model.
    """
    trainer = Trainer.from_config(train_config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    working_dir = Path(
        "/media/ceph-niu/neuroinformatics/sirmpilatzen/SLEAP_tutorial"
    )
    models_dir = working_dir / "models"
    labels_path = working_dir / "labels.v002.slp"
    skeleton_path = working_dir / "skeleton_v002.json"
    anchor_part = "centre"
    run_name_prefix = "EPM_topdown_v002_"
    train_fraction = 0.8

    labels_split_paths = split_labels(labels_path.as_posix(), train_fraction)

    centroid_cfg = configure_model(
        "centroid",
        labels_split_paths,
        skeleton_path=None,
        anchor_part=anchor_part,
        input_scaling=0.5,
        n_epochs=50,
        batch_size=4,
        run_name_prefix=run_name_prefix,
        models_dir=models_dir.as_posix(),
    )
    train_model(centroid_cfg)

    instance_cfg = configure_model(
        "centered_instance",
        labels_split_paths,
        skeleton_path=skeleton_path,
        anchor_part=anchor_part,
        input_scaling=1.0,
        n_epochs=100,
        batch_size=4,
        run_name_prefix=run_name_prefix,
        models_dir=models_dir.as_posix(),
    )
    train_model(instance_cfg)
