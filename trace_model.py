"""
trace a model to C++ readable
in .pt format

"""

from models.with_mobilenet import PoseEstimationWithMobileNet
from alfred.dl.torch.common import device
from alfred.utils.log import logger as logging
import torch
from models.utils import load_state


model_path = 'weights/checkpoint_iter_370000.pth'

model = PoseEstimationWithMobileNet(is_train=False).to(device)
load_state(model, torch.load(model_path))
logging.info('Pose estimator model loaded.')

example = torch.rand(1, 3, 256, 456).to(device)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save('human_pose_light_model.pt')
