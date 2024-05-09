import torch
from lib.HDSRNet.HDSRNet import utility as hdsrnet_utils
from lib.HDSRNet.HDSRNet import model as hdsrnet_model
from lib.HDSRNet.HDSRNet.option import args as hdsrnet_args
from modules.shared import SCALE
from modules.logger import logger

def init_hdsrnet():
    """
    Initialize the HDSRNet model with pre-configured settings and checkpoint.

    Returns:
        Model: The initialized HDSRNet model.
    """
    logger.debug('Init HDSRNet')
    torch.set_grad_enabled(False)
    #torch.manual_seed(args.seed)
    checkpoint = hdsrnet_utils.checkpoint(hdsrnet_args)
    model = hdsrnet_model.Model(hdsrnet_args, checkpoint)
    return model
def hdsrnet_upscale(model, img):
    """
    Upscale an image using the HDSRNet model.

    Args:
        model (Model): HDSRNet model
        img (Tensor): The input image tensor.

    Returns:
        Tensor: The upscaled and quantized image tensor.
    """
    model.eval()
    img = img.unsqueeze(0).cuda()
    res = model(img, SCALE)
    res = hdsrnet_utils.quantize(res, 255)
    img = res[0].byte()
    return img