import os
import torch
# Update basicsr model registry by importing hat_model
try:
    from lib.HAT.hat.models import hat_model
except:
    pass
from modules.logger import logger
from basicsr.models import build_model
from basicsr.utils import (
    get_env_info, get_root_logger, get_time_str, make_exp_dirs)
from basicsr.utils.options import dict2str
from modules.options import parse_options
from modules.shared import HAT_ROOT_PATH, HAT_OPTIONS_PATH

def init_hat():
    """
    Initialize the HAT model by parsing options, setting up the environment, and building the model.

    Returns:
        Model: The initialized HAT model.
    """
    logger.debug('Init HAT')
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(HAT_ROOT_PATH, HAT_OPTIONS_PATH, is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    hat_model = build_model(opt)
    return hat_model

def hat_upscale(hat_model, img):
    """
    Upscale an image using the HAT model.

    Args:
        hat_model (Model): HAT model
        img (Tensor): The input image tensor.

    Returns:
        Tensor: The upscaled image tensor.
    """
    img = hat_model.single_val(img.cuda(), save_img=False)[:, :, ::-1]
    img = img.copy()
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img