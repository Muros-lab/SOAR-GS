# SOAR-GS

**Semantically Guided Active Scene Reconstruction Using Gaussian Splatting and Class-Level Geometric Complexity**

This repository contains the code to run SOAR-GS in the Habitat simulator on Replica and Matterport3D scenes. It is built on [ActiveGS](https://github.com/dmar-bonn/active-gs).

## Requirements

- Ubuntu 20.04/22.04
- NVIDIA GPU (tested on an RTX 4090), CUDA 11.8 toolkit (`nvcc`), gcc 11 or older
- conda
- System packages:

```bash
sudo apt install build-essential git wget unzip pigz libjpeg-dev libglm-dev libgl1 libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
```

## Installation

```bash
git clone https://github.com/Muros-lab/SOAR-GS.git
cd SOAR-GS
bash envs/build.sh
conda activate soar-gs
```

`envs/build.sh` creates the conda environment `soar-gs`, builds habitat-sim 0.2.4 into `simulator/habitat-sim`, and installs PyTorch 2.1.2 (CUDA 11.8) and the packages in `envs/requirements.txt`.

## Model weights

```bash
mkdir -p weights
wget -P weights https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip
unzip -j weights/efficient_sam_vits.pt.zip -d weights
```

The YOLO-World weights (`weights/yolov8x-worldv2.pt`) and the CLIP text encoder (`~/.cache/clip`) are downloaded automatically on the first run, so the first run needs internet access.

## Datasets

The datasets are not included in this repository.

### Replica

Run from the repository root:

```bash
bash data/replica_download.sh   # all Replica scenes
bash data/replica_example.sh    # office_0 only
```

Expected structure:

```
data/replica_v1/
├── office_0/
│   ├── mesh.ply
│   └── habitat/replicaSDK_stage.stage_config.json
├── office_1/
│   ...
└── room_2/
```

Scene configs: `replica/office0` to `replica/office4` and `replica/room0` to `replica/room2` (`config/scene/replica/`).

### Matterport3D

Matterport3D is available on request from the dataset authors ([Matterport3D](https://niessner.github.io/Matterport/)). We used the habitat scenes (`mp3d_habitat`) and the scan meshes (`matterport_mesh`) of four scenes: `GdvgFV5R1Z5`, `gZ6f7yhEvPG`, `HxpKQynjfin`, and `pLe4wQe7qrG`. The habitat scene is used by the simulator, and the scan mesh only for the scene bounding box. Expected structure (the exact mesh paths are listed in `config/scene/mp3d/<scene>.yaml`):

```
data/MP3D/v1/tasks/mp3d/<scene>/<scene>.glb
data/mp3d_data/v1/scans/<scene>/<scene>/matterport_mesh/<id>/<id>.obj
```

Scene configs: `mp3d/GdvgFV5R1Z5`, `mp3d/gZ6f7yhEvPG`, `mp3d/HxpKQynjfin`, and `mp3d/pLe4wQe7qrG` (`config/scene/mp3d/`).

## Run

All commands are run from the repository root with the `soar-gs` environment activated.

Replica (420 s mission, seed 42, no GUI by default):

```bash
python main.py scene=replica/office0
```

Other scene and seed (use a new `experiment.run_id` for each run):

```bash
python main.py scene=replica/room0 seed=142 experiment.run_id=s142
```

With the live GUI (needs a display):

```bash
python main.py scene=replica/office0 use_gui=true
```

Matterport3D (900 s mission):

```bash
python main.py --config-name main_mp3d scene=mp3d/GdvgFV5R1Z5
```

Other settings are changed in the same way, for example `experiment.budget=300` for the mission time in seconds.

### Output

Results are written to `experiments/<exp_id>/<scene>/object_confidence/<run_id>/` (by default `experiments/test/replica/office0/object_confidence/0/`):

- `exp_config.yaml`: configuration of the run
- `map/map_XXX.th`: Gaussian maps saved during the mission
- `map/record_info.txt`: map index, mission time (s), and path length (m) of each saved map
- `global_path.pkl`: camera path

To view a saved map and the camera path:

```bash
python visualize.py -G experiments/test/replica/office0/object_confidence/0/map/map_XXX.th -C experiments/test/replica/office0/object_confidence/0/global_path.pkl
```

The mission time is the sum of the measured mapping and planning time and the travel time, so the amount of reconstruction within a mission depends on the GPU. Run one mission at a time.

## License and acknowledgements

This code is released under the MIT License (see `LICENSE`). It is based on [ActiveGS](https://github.com/dmar-bonn/active-gs) (MIT License), and `visualization/gl_render` is based on code by Li Ma (MIT License, see `visualization/gl_render/LICENSE`). The detector uses [YOLO-World](https://github.com/ultralytics/ultralytics) (AGPL-3.0) and [EfficientSAM](https://github.com/yformer/EfficientSAM) (Apache-2.0), which are installed as separate packages under their own licenses. The Replica and Matterport3D datasets are subject to their own terms and are not included.
