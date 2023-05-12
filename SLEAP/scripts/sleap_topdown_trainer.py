import copy
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


class SLEAPTrainer_TopDown_SingleInstance:
    def __init__(
        self,
        train_dir: Path,
        labels_path: Path,
        skeleton_path: Optional[Path] = None,
        train_fraction: float = 0.8,
        run_name_prefix: Optional[str] = None,
        anchor_part: str = "centre",
        camera_view: str = "top",
    ):
        """Class for training top-down SLEAP models on single instance data.
        It is inspired by a Jupyter Notebook written by Chang Huan Lo.

        Parameters
        ----------
        train_dir : pathlib.Path
            This is the directory where the training job will be run. It will
            be created if it doesn't exist.
        labels_path : pathlib.Path
            Path to SLEAP project file containing labeled data.
        skeleton_path : pathlib.Path, optional
            Path to skeleton file, by default None
        train_fraction : float, optional
            Fraction of labels to use for training set, by default 0.8
        run_name_prefix : str, optional
            Prefix to prepend to run name, by default None
        anchor_part : str, optional
            Name of part to use as anchor for instance cropping, by default "centre"
            Must be one of the part names in the skeleton.
        camera_view: str, optional
            Camera view for training data, by default "top".
            Must be one of "top", "bottom", or "side".
        """
        self.train_dir = train_dir
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.labels_path = labels_path
        self.skeleton_path = skeleton_path

        self.train_fraction = train_fraction
        self.run_name_prefix = run_name_prefix
        self.anchor_part = anchor_part

        self.cfg = TrainingJobConfig()

        self._split_labels()
        self._update_config_labels()
        self._update_config_outputs()

        if camera_view in ["top", "bottom", "side"]:
            self.camera_view = camera_view
        else:
            raise ValueError("camera_view must be one of 'top', 'bottom', or 'side'.")
        self._update_config_rotation()

        self.cfg.data.preprocessing.ensure_grayscale = True
        self.cfg.data.instance_cropping.center_on_part = anchor_part
        self.cfg.data.instance_cropping.crop_size_detection_padding = 32
        self.cfg.data.instance_cropping.crop_size = 350         

    def _split_labels(self):
        """Load labels from an exported SLEAP training-job labels package,
        split into train/val/test sets,
        and save as separate SLEAP package files. The split files will be saved
        in the training job directory `train_dir`, with the split name appended
        to the file name (e.g., "my_labels.pkg.slp" -> "my_labels.train.pkg.slp")

        Creates the following attribute:
        - `labels_split_paths`: dictionary with keys "train", "val", and "test"
        """
        labels = sleap.load_file(self.labels_path.as_posix())
        labels_train, labels_val_test = labels.split(n=self.train_fraction)
        labels_val, labels_test = labels_val_test.split(n=0.5)
        labels_split_data = {
            "train": labels_train,
            "val": labels_val,
            "test": labels_test,
        }
        labels_split_paths = {}
        for split, labels in labels_split_data.items():
            labels_split_name = self.labels_path.name.replace(
                ".pkg.slp", f".{split}.pkg.slp"
            )
            labels_split_path = self.train_dir / labels_split_name
            labels.save(labels_split_path.as_posix(), with_images=True)
            labels_split_paths[split] = labels_split_path.as_posix()

        self.labels_split_paths = labels_split_paths

    def _update_config_labels(self):
        """Update training job configuration with paths to split labels."""
        self.cfg.data.labels.training_labels = self.labels_split_paths["train"]
        self.cfg.data.labels.validation_labels = self.labels_split_paths["val"]
        self.cfg.data.labels.test_labels = self.labels_split_paths["test"]

    def _update_config_outputs(self):
        """Update training job configuration with model outputs options."""
        self.cfg.outputs.run_name_prefix = self.run_name_prefix
        self.cfg.outputs.runs_folder = (self.train_dir / "models").as_posix()
        self.cfg.outputs.save_outputs = True
        self.cfg.outputs.save_visualizations = True
        self.cfg.outputs.checkpointing.initial_model = True
        self.cfg.outputs.checkpointing.best_model = True

    def _update_config_rotation(self):
        """Update training job configuration with suitable rotation
        augmentation options, depending on the camera view.
        """
        self.cfg.optimization.augmentation_config.rotate = True

        if self.camera_view in ["top", "bottom"]:
            self.cfg.optimization.augmentation_config.rotation_min_angle = -180
            self.cfg.optimization.augmentation_config.rotation_max_angle = 180
        elif self.camera_view == "side":
            self.cfg.optimization.augmentation_config.rotation_min_angle = -30
            self.cfg.optimization.augmentation_config.rotation_max_angle = 30

    def configure_model(
        self,
        model_type: str = "centroid",
        input_scaling: float = 0.5,
        n_epochs: int = 50,
        batch_size: int = 4,
        max_stride: int = 32,
    ):
        """Configure training job for model training.

        Parameters
        ----------
        model_type : str
            Type of model to train. Must be one of "centroid" or "centered_instance".
            Default is "centroid".
        input_scaling : float
            Scaling factor for input images. Default is 0.5.
        n_epochs : int
            Number of training epochs. Default is 50.
        batch_size : int
            Batch size for training. Default is 4.
        max_stride : int
            Maximum stride for UNet backbone. Default is 32.
        """

        self.model_type = model_type
        self.cfg.data.preprocessing.input_scaling = input_scaling
        self.cfg.optimization.epochs = n_epochs
        self.cfg.optimization.batch_size = batch_size
        
        if self.model_type == "centroid":
            self.cfg.outputs.run_name_suffix = ".centroid"
            self.cfg.model.backbone.unet = UNetConfig(
                max_stride=max_stride,
                filters=16,
                filters_rate=2.00,
                output_stride=2,
                up_interpolate=True,
            )
            self.cfg.model.heads.centroid = CentroidsHeadConfig(
                anchor_part=self.anchor_part, sigma=2.5, output_stride=2
            )
            self.cfg.model.heads.centered_instance = None

        elif self.model_type == "centered_instance":
            self.cfg.outputs.run_name_suffix = ".centered_instance"
            skeleton = sleap.Skeleton.load_json(self.skeleton_path.as_posix())
            self.cfg.data.labels.skeletons = [skeleton]

            self.cfg.model.backbone.unet = UNetConfig(
                max_stride=max_stride,
                filters=24,
                filters_rate=2.00,
                output_stride=4,
                up_interpolate=True,
            )
            self.cfg.model.heads.centered_instance = (
                CenteredInstanceConfmapsHeadConfig(
                    anchor_part=self.anchor_part,
                    sigma=2.5,
                    output_stride=4,
                    loss_weight=1.0,
                )
            )
            self.cfg.model.heads.centroid = None
        
        else:
            raise ValueError(
                "model_type must be one of 'centroid' or 'centered_instance'."
            )

    def train_model(self):
        """Train SLEAP model using the given training configurations."""
        trainer = Trainer.from_config(self.cfg)
        trainer.setup()
        trainer.train()
