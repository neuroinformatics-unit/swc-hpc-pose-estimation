import argparse
from pathlib import Path
from sleap_topdown_trainer import SLEAPTrainer_TopDown_SingleInstance

TEST_DATA_DIR = Path("/ceph/scratch/neuroinformatics-dropoff/SLEAP_HPC_test_data")


def main(batch_size=4):
    """Train SLEAP model with variable batch size.

    This functions runs a SLEAP training job using the
    SLEAPTrainer_TopDown_SingleInstance class from
    `sleap_topdown_trainer.py`, which should be located in the same
    directory as this script. SLEAPTrainer_TopDown_SingleInstance
    is a convenience class for training top-down SLEAP models on
    single instance data.
    
    Parameters
    ----------
    batch_size : int
        Batch size for training. Default is 4.
        Applies to both centroid and centered instance models
        of the top-down configuration.
    """
    training_job = SLEAPTrainer_TopDown_SingleInstance(
        train_dir=TEST_DATA_DIR / "labels.v002.slp.array_training",
        labels_path=TEST_DATA_DIR / "labels.v002.slp.training_job/labels.v002.pkg.slp",
        skeleton_path=TEST_DATA_DIR / "skeleton.json",
        train_fraction=0.8,
        run_name_prefix=f"batch-size-{batch_size}",
        anchor_part="centre",
        camera_view="top",
    )
    training_job.configure_centroid_model(
        input_scaling=0.5,
        n_epochs=50,
        batch_size=batch_size,
        max_stride=32,
    )
    training_job.configure_centered_instance_model(
        input_scaling=1.0,
        n_epochs=100,
        batch_size=batch_size,
        max_stride=32,
    )
    training_job.train_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SLEAP model with variable batch size."
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        "-b",
        type=int,
        default=4,
        dest="batch_size",
        help="Batch size for training (default: 4)",
    )
    args = parser.parse_args()
    main(batch_size=args.batch_size)
