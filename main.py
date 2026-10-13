import hydra
import torch
import warnings
import torch.multiprocessing as mp
import os
import yaml
import numpy as np
import random
from omegaconf import OmegaConf

from utils.common import MissionRecorder
from simulator import get_simulator
from mapping import get_mapper
from planning import get_planner


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


warnings.simplefilter("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def main(cfg):
    # Set seed for reproducibility
    seed = cfg.get('seed', 42)
    set_seed(seed)
    print(f"[Main] Random seed set to {seed}")
    
    # set up mode config
    if cfg.debug:
        mission_recorder = None

    else:
        experiment_path = os.path.join(
            cfg.experiment.output_dir,
            str(cfg.experiment.exp_id),
            cfg.scene.scene_name,
            cfg.planner.planner_name,
            str(cfg.experiment.run_id),
        )
        os.makedirs(experiment_path, exist_ok=True)

        # save experiment configuration
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(f"{experiment_path}/exp_config.yaml", "w") as file:
            yaml.dump(cfg_dict, file)

        mission_recorder = MissionRecorder(experiment_path, cfg.experiment)

    # load components
    mapping_agent = get_mapper(cfg, device)
    simulator = get_simulator(cfg)
    planner = get_planner(cfg, device)

    # set up gui messages
    mp.set_start_method("spawn")
    if cfg.use_gui:
        # imported only when the GUI is used, so headless runs do not need the OpenGL/GUI libraries
        from visualization import gui

        init_event = mp.Event()
        q_mapper2gui = mp.Queue()
        q_gui2mapper = mp.Queue()
        q_planner2gui = mp.Queue()
        q_gui2planner = mp.Queue()

        mapping_agent.use_gui = True
        mapping_agent.q_mapper2gui = q_mapper2gui
        mapping_agent.q_gui2mapper = q_gui2mapper

        planner.q_planner2gui = q_planner2gui
        planner.q_gui2planner = q_gui2planner

        params_gui = {
            "mapper_receive": q_mapper2gui,
            "mapper_send": q_gui2mapper,
            "planner_receive": q_planner2gui,
            "planner_send": q_gui2planner,
        }
        gui_process = mp.Process(
            target=gui.run,
            args=(init_event, cfg.gui, params_gui),
        )
        gui_process.start()
        while not init_event.wait(1.0):
            if not gui_process.is_alive():
                raise RuntimeError("GUI process failed to start (no display?). Run with use_gui=false.")

    # load components to mapping module
    mapping_agent.load_recorder(mission_recorder)
    mapping_agent.load_simulator(simulator)
    mapping_agent.load_planner(planner)

    # start mission
    mapping_agent.run()

    # wait for gui to be closed
    if cfg.use_gui:
        gui_process.join()


if __name__ == "__main__":
    main()
