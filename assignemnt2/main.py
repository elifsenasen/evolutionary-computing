from controller import inputs_of_controller,genotype_length,genotype_to_weights,angle_of_controller 

from pathlib import Path
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec import set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder


SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)
hidden_size = 8
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation


#model and data should be created on run_experiment()
input_size = inputs_of_controller(data, TARGET_POSITION).size
output_size = model.nu
genes_number = genotype_length(input_size, hidden_size, output_size)

weights=genotype_to_weights(np.random.rand(genes_number), input_size, hidden_size, output_size)
