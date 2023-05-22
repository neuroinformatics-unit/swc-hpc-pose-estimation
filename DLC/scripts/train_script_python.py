from pathlib import Path
import deeplabcut as dlc

# Project info
task = 'EPM'
experimenter = 'NS'

project_dir = Path('/ceph/neuroinformatics/neuroinformatics/sirmpilatzen/DLC_HPC_test_data')

# Paths to videos
videos_dir = project_dir / 'videos'
video_names = ['M708149_EPM_20200317_165049331-converted.mp4']
video_paths = [videos_dir / video_name for video_name in video_names]
for video_path in video_paths:
    assert video_path.is_file(), f'Video {video_path} does not exist'
video_paths_str = [video_path.as_posix() for video_path in video_paths]

"""
path_config_file=dlc.create_new_project(
    project=task,
    experimenter=experimenter,
    videos=video_paths_str,
    working_directory=project_dir.as_posix(),
    copy_videos=False,
    multianimal=False,
)
"""

path_config_file = project_dir / f"{task}-{experimenter}-2023-05-22"/ "config.yaml"

# Edit config file
config = dlc.auxiliaryfunctions.read_config(path_config_file)
config['scorer'] = 'NS'
config['TrainingFraction'] = [0.8]
config['bodyparts'] = [
    'snout',
    'left_ear',
    'right_ear',
    'tail_base',
    ]
config['skeleton'] = [
    ['snout', 'left_ear'],
    ['snout', 'right_ear'],
    ['snout', 'tail_base']
]
config['numframes2pick'] = 20
dlc.auxiliaryfunctions.write_config(path_config_file, config)

# Extract frames
dlc.extract_frames(
    path_config_file,
    mode='automatic',
    algo='uniform',
    userfeedback=False
)

# Label using local machine GUI and save labels

# Create training dataset
dlc.create_training_dataset(path_config_file)

# Train network
dlc.train_network(
    path_config_file,
    shuffle=1,
    saveiters=1000,
    displayiters=100,
    maxiters=10000)
