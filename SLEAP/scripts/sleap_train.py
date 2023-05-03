import sleap
from pathlib import Path

from sleap.nn.config import (
    TrainingJobConfig,
    UNetConfig,
    CentroidsHeadConfig,
    CenteredInstanceConfmapsHeadConfig,
)
from sleap.nn.training import Trainer

working_dir = Path("/media/ceph-niu/neuroinformatics/sirmpilatzen/SLEAP_tutorial")
data_dir = working_dir / "videos"
models_dir = working_dir / "models"

version = "v002"
train_fraction = 0.8
anchor_part = "centre"
labels_path = working_dir / f"labels.{version}.slp"
skeleton_path = working_dir / f"skeleton_{version}.json"

run_name = f"EPM_topdown_labels-{version}"
runs_folder = models_dir

################################################
# Set up training data
################################################

# Split labels into train (0.8) - validation (0.1) - test (0.1) sets
labels_path = labels_path.as_posix()
labels = sleap.load_file(labels_path)
labels_train, labels_val_test = labels.split(n=train_fraction)
labels_val, labels_test = labels_val_test.split(n=0.5)
labels_split_data = {"train": labels_train, "val": labels_val, "test": labels_test}
# Save as separate files (including images)
for split, labels in labels_split_data.items():
    labels_split_path = labels_path.replace(".slp", f".{split}.pkg.slp")
    labels.save(labels_split_path, with_images=True)


################################################
# Configure and train centroid model
################################################

# Initalise default training config
centroid_cfg = TrainingJobConfig()

# Add paths to labels
centroid_cfg.data.labels.training_labels = labels_path.replace(".slp", ".train.pkg.slp")
centroid_cfg.data.labels.validation_labels = labels_path.replace(".slp", ".val.pkg.slp")
centroid_cfg.data.labels.test_labels = labels_path.replace(".slp", ".test.pkg.slp")

# Data preprocessing, augmentation, and optimisation parameters
centroid_cfg.data.preprocessing.input_scaling = 0.5
centroid_cfg.data.preprocessing.ensure_grayscale = True
centroid_cfg.data.instance_cropping.center_on_part = anchor_part
centroid_cfg.data.instance_cropping.crop_size_detection_padding = 32
centroid_cfg.optimization.augmentation_config.rotate = True
centroid_cfg.optimization.augmentation_config.rotation_min_angle = -180
centroid_cfg.optimization.augmentation_config.rotation_max_angle = 180
centroid_cfg.optimization.epochs = 50
centroid_cfg.optimization.batch_size = 4

# configure NN backbone (UNet) and head (centroid)
centroid_cfg.model.backbone.unet = UNetConfig(
    max_stride=16,
    filters=16,
    filters_rate=2.00,
    output_stride=2,
    up_interpolate=True,
)
centroid_cfg.model.heads.centroid = CentroidsHeadConfig(
    anchor_part=anchor_part, sigma=2.5, output_stride=2
)

# configure outputs
centroid_cfg.outputs.run_name = run_name
centroid_cfg.outputs.save_outputs = True
centroid_cfg.outputs.runs_folder = runs_folder.as_posix()
centroid_cfg.outputs.save_visualizations = True
centroid_cfg.outputs.checkpointing.initial_model = True
centroid_cfg.outputs.checkpointing.best_model = True

# Run training
centroid_trainer = Trainer.from_config(centroid_cfg)
centroid_trainer.setup()
centroid_trainer.train()

###############################################
# Configure and train centered instance model
###############################################

skeleton = sleap.Skeleton.load_json(skeleton_path.as_posix())  # load the skeleton

# initalise default training job config
instance_cfg = TrainingJobConfig()

# Add paths to labels and skeleton
instance_cfg.data.labels.training_labels = labels_path.replace(".slp", ".train.pkg.slp")
instance_cfg.data.labels.validation_labels = labels_path.replace(".slp", ".val.pkg.slp")
instance_cfg.data.labels.test_labels = labels_path.replace(".slp", ".test.pkg.slp")
instance_cfg.data.labels.skeletons = [skeleton]

# Data preprocessing, augmentation, and optimisation parameters
instance_cfg.data.preprocessing.input_scaling = 1.0
instance_cfg.data.preprocessing.ensure_grayscale = True
instance_cfg.data.instance_cropping.center_on_part = anchor_part
instance_cfg.data.instance_cropping.crop_size_detection_padding = 32
instance_cfg.optimization.augmentation_config.rotate = True
instance_cfg.optimization.augmentation_config.rotation_min_angle = -180
instance_cfg.optimization.augmentation_config.rotation_max_angle = 180
instance_cfg.optimization.epochs = 100
instance_cfg.optimization.batch_size = 4

# configure NN backbone (UNet) and head (centered instance)
# configure NN backbone (UNet) and head (centroid)
instance_cfg.model.backbone.unet = UNetConfig(
    max_stride=16,
    filters=24,
    filters_rate=2.0,
    output_stride=4,
    up_interpolate=True,
)
instance_cfg.model.heads.centered_instance = CenteredInstanceConfmapsHeadConfig(
    anchor_part=anchor_part,
    sigma=2.5,
    output_stride=4,
    loss_weight=1.0,
)

# configure outputs
instance_cfg.outputs.run_name = run_name
instance_cfg.outputs.save_outputs = True
instance_cfg.outputs.runs_folder = runs_folder.as_posix()
instance_cfg.outputs.save_visualizations = True
instance_cfg.outputs.checkpointing.initial_model = True
instance_cfg.outputs.checkpointing.best_model = True

# Run training
instance_trainer = Trainer.from_config(instance_cfg)
instance_trainer.setup()
instance_trainer.train()
