import sys
import os
sys.path.append(os.path.abspath('./lib'))
from modules.codec import EncFFMPEG
import modules.shared as shared
import math
import gradio as gr
import numpy as np
import torch
from PIL import Image
from modules.shared import outputpath
try:
    from lib.HAT.hat.models import hat_model
except:
    pass
import logging
import torch
from os import path as osp
import torchvision.transforms as transforms
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str
from modules.options import parse_options
from modules.shared import models_path, models_weights
from lib.HDSRNet.HDSRNet import utility
from lib.HDSRNet.HDSRNet import model
from lib.HDSRNet.HDSRNet.option import args

mode = 0
upscaling_resize_multiplier = 4
container_constructor = EncFFMPEG
container = None
hat_model = None
hdsrnet_model = None
opt_path = 'lib/HAT/options/test/HAT_SRx4_ImageNet-pretrain.yml'
root_path = 'lib/HAT'


def init_hdsrnet():
    print('Init HDSRNet')
    torch.set_grad_enabled(False)
    #torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    hdsrnet_model = model.Model(args, checkpoint)
    return hdsrnet_model

def init_hat():
    print('Init HAT')
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, opt_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    hat_model = build_model(opt)
    return hat_model

def hat_upscale(img):
    img = hat_model.single_val(img.cuda(), save_img=False)[:, :, ::-1]
    img = img.copy()
    img = torch.from_numpy(img).permute(2, 0, 1)
    return img

def hdsrnet_upscale(img):
    hdsrnet_model.eval()
    img = img.unsqueeze(0).cuda()
    res = hdsrnet_model(img, upscaling_resize_multiplier)
    res = utility.quantize(res, 255)
    img = res[0].byte().cpu()
    return img

def upscale(img, scale: int, selected_model: str = None):
    scale = scale
    img = img.permute(2, 0, 1).float()
    if selected_model == 'HAT':
        img = hat_upscale(img)
    elif selected_model == 'HDSRNet':
        img = hdsrnet_upscale(img)
    img = (img.permute(1,2,0).cpu().numpy()).round().astype(np.int8)
    return img

def uniform_temporal_subsample(x: torch.Tensor, num_samples: int, temporal_dim: int = -3) -> torch.Tensor:
    """
    Uniformly subsamples num_samples indices from the temporal dimension of the video.
    When num_samples is larger than the size of temporal dimension of the video, it
    will sample frames based on nearest neighbor interpolation.
    Args:
        x (torch.Tensor): A video tensor with dimension larger than one with torch
            tensor type includes int, long, float, complex, etc.
        num_samples (int): The number of equispaced samples to be selected
        temporal_dim (int): dimension of temporal to perform temporal subsample.
    Returns:
        An x-like Tensor with subsampled temporal dimension.
    """
    
    if type(x) == np.ndarray:
        x = x.transpose(3,0,1,2)
        x=torch.from_numpy(x)
    else:
        x = x.permute(3,0,1,2)
    t = x.shape[temporal_dim]
    assert num_samples > 0 and t > 0
    # Sample by nearest neighbor interpolation if num_samples > t.
    indices = torch.linspace(0, t - 1, num_samples)
    indices = torch.clamp(indices, 0, t - 1).long()
    return torch.index_select(x, temporal_dim, indices)

def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    """
    Determines the shorter spatial dim of the video (i.e. width or height) and scales
    it to the given size. To maintain aspect ratio, the longer side is then scaled
    accordingly.
    Args:
        x (torch.Tensor): A video tensor of shape (C, T, H, W) and type torch.float32.
        size (int): The size the shorter side is scaled to.
        interpolation (str): Algorithm used for upsampling,
            options: nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
    Returns:
        An x-like Tensor with scaled spatial dims.
    """
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))

    return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=False)

def inference_step(vid, duration, out_fps, upscaler):
    video_arr, audio_arr = vid.decode(duration)
    x = uniform_temporal_subsample(video_arr, duration * out_fps)
    x = x.permute(1, 2, 3, 0)
    dest_w = x.shape[2] * upscaling_resize_multiplier
    dest_h = x.shape[1] * upscaling_resize_multiplier
    buffer = np.empty((x.shape[0], dest_h, dest_w, x.shape[3]),dtype=np.uint8) # [b, h, w, c]
    with torch.no_grad():
        for index in range(0, x.shape[0]):
            output = x[index]
            buffer[index] = upscale(output, upscaling_resize_multiplier, upscaler)

    return buffer, audio_arr

def predict_fn(filepath, start_sec, end_sec, out_fps=-1, upscaler='HDSRNet'):
    global container_constructor, container, hdsrnet_model, hat_model
    if upscaler == 'HDSRNet':
        hdsrnet_model = init_hdsrnet()
    else:
        hat_model = init_hat()
    frame_len = 1
    name = os.path.splitext(os.path.basename(filepath))[0]

    container = container_constructor(filepath, outputpath+name+'.mp4', width=-1, height=-1, out_fps=out_fps, 
                                        start_time=start_sec, duration=end_sec - start_sec)
    if out_fps == -1:
        out_fps = container.video_fps()

    for b in  range(0, end_sec - start_sec, shared.MAX_DURATION_BATCH):
        b_start = b + start_sec
        clip_duration = shared.MAX_DURATION_BATCH if b_start+shared.MAX_DURATION_BATCH<=end_sec else end_sec - b_start
        # container.reset(b_start, clip_duration)
        msg = "🎠 Clip {}s - {}s".format(b_start, b_start+clip_duration)
        print(msg)
        for i in range(0, shared.MAX_DURATION_BATCH if b_start+shared.MAX_DURATION_BATCH<=end_sec else end_sec - b_start, frame_len):
            if b_start + i > container.total_duration:
                break
            print(f"🖼️ Processing step {i + 1}/{shared.MAX_DURATION_BATCH if b_start+shared.MAX_DURATION_BATCH<=end_sec else end_sec - b_start}...")
            video, audio = inference_step(vid=container, duration=frame_len, out_fps=out_fps, upscaler=upscaler)
            
            if i == 0:
                video_all = np.zeros([out_fps*clip_duration, video.shape[1], video.shape[2], video.shape[3]], np.uint8)
                audio_all = audio
                
            video_all[i*out_fps: (i+1)*out_fps] = video
            
        print(f"💾 Writing output video...")
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        container.remux(video_all, audio_all)
        
        del audio_all
        del video_all
        # gc.collect()
    container.close()
    print(f"✅ Done!")
    return outputpath+name+'.mp4'

def on_video_change(filepath, start_sec, out_fps):
    global container
    if filepath is None:
        return 
    name = os.path.splitext(os.path.basename(filepath))[0]
    container = container_constructor(filepath, outputpath+name+'.mp4', width=-1, height=-1, out_fps=out_fps, start_time=0, duration=1)
    max_duration = container.get_total_duration()
    container.get_stats()
    stats = [ (key, value) for key,value in container.get_stats()['video'].items()]
    stats = ""
    stats += '**video:**  \n    '
    for key,value in container.get_stats()['video'].items():
        stats += "   {}: {}".format(key, value)
    stats += '  \n  **audio:**  \n    '
    for key,value in container.get_stats()['audio'].items():
        stats += " {}:{}".format(key, value)
    del container
    return [gr.Slider.update(value=start_sec if start_sec<max_duration else 0, maximum=max_duration), 
            gr.Slider.update(value=round(max_duration) , maximum=max_duration),\
            stats
            ]

def main_tab():
    with gr.Row():
        with gr.Column():
            inputs_video = gr.Video(source="upload", interactive=True)
            stats = gr.Markdown()

            with gr.Group() as meta:
                start_sec = gr.Slider(minimum=0, maximum=3000, step=1, value=0, label="start_sec", interactive=True)
                end_sec = gr.Slider(minimum=1, maximum=3000, step=1, value=6, label="end_sec", interactive=True)
            with gr.Group() as video_setup:
                out_fps = gr.Slider(minimum=-1, maximum=30, step=1, value=24, label="output_fps (original fps: -1)", interactive=True)

            with gr.Group() as upscaler_setup:
                upscaler = gr.Radio(choices=['HDSRNet','HAT'], label="Upscaler",value='HDSRNet', interactive=True)

            with gr.Group() as control:
                with gr.Row():
                    submit = gr.Button("Submit")
    
        outputs_video = gr.Video(interactive=False)
    examples = gr.Examples([
        ['./assets/examples/video_example_1_photo_240p.mp4', 0, 1],
        ['./assets/examples/video_example_2_bone_240p.mp4', 0, 6]
    ], inputs=[inputs_video, start_sec, end_sec], outputs=[inputs_video, start_sec, end_sec])
    submit.click(fn=predict_fn, inputs=[inputs_video, start_sec, end_sec, out_fps, upscaler], outputs=[outputs_video])
    inputs_video.change(fn=on_video_change, inputs=[inputs_video, start_sec, out_fps], outputs=[start_sec, end_sec, stats])

with gr.Blocks(title="Video Super Resolution") as app:
    gr.Label("Video Super Resolution")
    with gr.Tab(label="Super Resolution"):
        main_tab()


if __name__ == '__main__':
    app.launch()