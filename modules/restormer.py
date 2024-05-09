import os
import torch
import torch.nn.functional as F
import wget
from runpy import run_path
import gradio as gr
from modules.logger import logger
from modules.shared import RESTORMER_WEIGHTS, RESTORMER_PRETRAINED_PATHS, RESTORMER_BASE_MODELS_PATH

def init_restormer(restormer_task):
    """
    Sets up the Restormer model for video enhancement

    Args:
        restormer_task (str) Restormer task: Denoise, Deblur, Deraining

    Returns:
        Model: The initialized Restormer model.
    """
    parameters = {
        'inp_channels': 3,
        'out_channels': 3,
        'dim': 48,
        'num_blocks': [4, 6, 6, 8],
        'num_refinement_blocks': 4,
        'heads': [1, 2, 4, 8],
        'ffn_expansion_factor': 2.66,
        'bias': False,
        'LayerNorm_type': 'WithBias',
        'dual_pixel_task': False
    }
    if restormer_task == 'Real Denoising':
        parameters['LayerNorm_type'] = 'BiasFree'

    # Get model weights and parameters
    model_path = RESTORMER_WEIGHTS[restormer_task]
    if not os.path.exists(model_path):
        gr.Info("Downloading Restormer model weights for {restormer_task}. Please wait...")
        logger.debug(f'Downloading Restormer model weights for {restormer_task} task to {RESTORMER_BASE_MODELS_PATH}')
        if not os.path.exists(RESTORMER_BASE_MODELS_PATH):
            os.mkdir(RESTORMER_BASE_MODELS_PATH)
        wget.download(RESTORMER_PRETRAINED_PATHS[restormer_task], out=RESTORMER_BASE_MODELS_PATH)
        gr.Info("✅ Download successful")
            
    load_arch = run_path(os.path.join('lib', 'Restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
    restormer_model = load_arch['Restormer'](**parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    restormer_model.to(device)

    checkpoint = torch.load(model_path)
    restormer_model.load_state_dict(checkpoint['params'])
    restormer_model.eval()

    logger.debug(f"\n ==> Running {restormer_task} with weights {model_path}\n")
    return restormer_model

def run_restormer(img, restormer_model, tile_size=None, tile_overlap=None):
    """
    Processes images using the Restormer model to enhance image quality, with options for tiling.

    Args:
        images (torch.Tensor): A batch of images as a 4D Tensor.
        restormer_model (Model): Restormer model object
        tile_size (int, optional): The size of the tiles to process the image in parts. Must be a multiple of 8.
                                   If None, the image is processed as a whole.
        tile_overlap (int, optional): The overlap between tiles when the image is processed in parts.

    Returns:
        torch.Tensor: The enhanced image as a 3D Tensor.
    """
    img_multiple_of = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = img.unsqueeze(0)
        input_ = img.float().div(255.).permute(0, 3, 1, 2).to(device)

        # Pad the input if not_multiple_of 8
        height, width = input_.shape[2], input_.shape[3]
        H = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
        W = ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - height if height % img_multiple_of != 0 else 0
        padw = W - width if width % img_multiple_of != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        if tile_size is None:
            # Testing on the original resolution image
            restored = restormer_model(input_)
        else:
            # test the image tile by tile
            b, c, h, w = input_.shape
            tile = min(tile_size, h, w)
            assert tile % 8 == 0, "tile size should be multiple of 8"
            tile_overlap = tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
            w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
            E = torch.zeros(b, c, h, w).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = restormer_model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)
                    E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
                    W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            restored = E.div_(W)

            restored = torch.clamp(restored, 0, 1)

            # Unpad the output
            restored = restored[:,:,:height,:width]

            restored = restored.permute(0, 2, 3, 1)# .cpu().detach().numpy()
            restored = (restored * 255.0).to(torch.uint8).cpu()

    return restored[0]