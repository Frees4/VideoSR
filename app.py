import os
import gradio as gr
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from PIL import Image
from modules.logger import logger
from modules.shared import (
    OUTPUT_PATH, MAX_DURATION_BATCH, SCALE, RESOLUTIONS)
from modules.utils import uniform_temporal_subsample
from modules.hdsrnet import init_hdsrnet, hdsrnet_upscale
from modules.hat import init_hat, hat_upscale
from modules.restormer import init_restormer, run_restormer
from modules.codec import EncFFMPEG

container_constructor = EncFFMPEG
container = None
hat_model = None
hdsrnet_model = None
restormer_model = None
should_continue = True


def upscale(img, selected_model=None, tile_size=None, tile_overlap=None):
    """
    Upscale an image x4 using either the HAT or HDSRNet model with optional tiling.

    Args:
        img (Tensor): The input image tensor.
        selected_model (str, optional): The model to use for upscaling. Options are 'HAT' or 'HDSRNet'.
        tile_size (int, optional): The size of the tiles to process the image in parts. Must be a multiple of 8.
        tile_overlap (int, optional): The overlap size between tiles.

    Returns:
        np.ndarray: The upscaled image as a numpy array with data type int8.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        input_ = img.float().permute(2,0,1)

        if tile_size is None:
            # Testing on the original resolution image
            input_ = input_.to(device)
            if selected_model not in ['HAT', 'HDSRNet']:
                raise ValueError("Invalid value for selected_model. Valid values are 'HAT' and 'HDSRNet'.")
            if selected_model == 'HAT':
                result = hat_upscale(hat_model, input_)
            elif selected_model == 'HDSRNet':
                result = hdsrnet_upscale(hdsrnet_model, input_)
        else:
            # test the image tile by tile
            c, h, w = input_.shape
            # If tile size > image shape, upscale whole image
            if tile_size > h or tile_size > w:
                input_ = input_.to(device)
                if selected_model == 'HAT':
                    result = hat_upscale(hat_model, input_)
                elif selected_model == 'HDSRNet':
                    result = hdsrnet_upscale(hdsrnet_model, input_)
                return (result.permute(1,2,0).cpu().numpy()).round().astype(np.int8)
                
            tile = min(tile_size, h, w)
            tile_overlap = tile_overlap

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(c, h*SCALE, w*SCALE).type_as(input_)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile].to(device)
                    if selected_model == 'HAT':
                        out_patch = hat_upscale(hat_model, in_patch).cpu()
                    elif selected_model == 'HDSRNet':
                        out_patch = hdsrnet_upscale(hdsrnet_model, in_patch).cpu()
                    out_patch_mask = torch.ones_like(out_patch)
                    E[..., h_idx*SCALE:(h_idx+tile)*SCALE, w_idx*SCALE:(w_idx+tile)*SCALE].add_(out_patch)
                    W[..., h_idx*SCALE:(h_idx+tile)*SCALE, w_idx*SCALE:(w_idx+tile)*SCALE].add_(out_patch_mask)
            result = E.div_(W)
        result = (result.permute(1,2,0).cpu().numpy()).round().astype(np.int8)
    return result


def inference_step(vid, duration, resolution, out_fps, upscaler, tile_size, tile_overlap, restormer_task):
    """
    Processes a video by decoding it, subsampling temporally, and applying upscaling and optional Restormer enhancement.

    Args:
        vid (VideoFileClip): The video file clip to be processed.
        duration (int): Duration in seconds to decode from the video.
        resolution (str): Desired resolution option
        out_fps (int): Output frames per second, used to determine the number of frames to subsample.
        upscaler (str): The name of the upscaling model to use ('HAT' or 'HDSRNet').
        tile_size (int): The size of the tiles to process the image in parts; must be a multiple of 8.
        tile_overlap (int): The overlap size between tiles.
        restormer_task (str): Specifies whether to use the Restormer model for enhancement ('Enabled' or 'Disabled').

    Returns:
        tuple: A tuple containing:
            - buffer (np.ndarray): An array of the processed video frames.
            - audio_arr (np.ndarray): An array containing the audio extracted from the video.
    """
    video_arr, audio_arr = vid.decode(duration)
    x = uniform_temporal_subsample(video_arr, duration * out_fps)
    x = x.permute(1, 2, 3, 0)
    b, h, w, c = x.shape
    if resolution != 'x4':
        dest_resolution = RESOLUTIONS[resolution]
        res_w, res_h = dest_resolution
        while (w, h) != (res_w, res_h) and h < res_h:
            dest_w = w * SCALE
            dest_h = h * SCALE
            buffer = np.empty((b, dest_h, dest_w, c),dtype=np.uint8) # [b, h, w, c]
            with torch.no_grad():
                for index in range(0, x.shape[0]):
                    output = x[index]
                    if restormer_task != 'Disabled':
                        output = run_restormer(output, restormer_model, tile_size, tile_overlap)
                    buffer[index] = upscale(output, upscaler, tile_size, tile_overlap)
            x = torch.from_numpy(buffer)
            b, h, w, c = x.shape
                    
        if (w, h) == (res_w, res_h):
            return x, audio_arr
        elif h >= res_h:
            result = np.empty((b, res_h, res_w, c),dtype=np.uint8) # [b, h, w, c]
            x = x.permute(0, 3, 1, 2)
            for i in range(len(x)):
                image = to_pil_image(x[i]) 
                width, height = image.size
                new_width = int(res_h * (width / height))
                new_height = res_h
                resized_image = image.resize((new_width, new_height))
                # Create a new blank image with black background
                new_image = Image.new("RGB", (res_w, res_h), (0, 0, 0))

                padding = (res_w - new_width) // 2

                # Paste the resized image onto the blank image with padding
                new_image.paste(resized_image, (padding, 0, padding + new_width, new_height))
                resized_tensor = pil_to_tensor(new_image).permute(1, 2, 0)
                result[i] = resized_tensor.cpu()
            return result, audio_arr


    dest_w = w * SCALE
    dest_h = h * SCALE
    buffer = np.empty((b, dest_h, dest_w, c),dtype=np.uint8) # [b, h, w, c]
    with torch.no_grad():
        for index in range(0, x.shape[0]):
            output = x[index]
            if restormer_task != 'Disabled':
                output = run_restormer(output, restormer_model, tile_size, tile_overlap)
            buffer[index] = upscale(output, upscaler, tile_size, tile_overlap)

    return buffer, audio_arr

def predict_fn(filepath, start_sec, end_sec, resolution, out_fps=0, upscaler='HDSRNet', tile_size=None, tile_overlap=None, restormer_task='Disabled', progress=gr.Progress()):
    """
    Processes a video file from a specified start to end time, applies upscaling and potentially other restoration tasks, and outputs the processed video.

    Args:
        filepath (str): Path to the input video file.
        start_sec (int): Start time in seconds for the video processing.
        end_sec (int): End time in seconds for the video processing.
        resolution (str): Desired resolution option
        out_fps (int, optional): Output frames per second. If set to 0, the original video's fps is used. Default is 0.
        upscaler (str, optional): Choice of upscaling model to use. Options are 'HDSRNet' or 'HAT'. Default is 'HDSRNet'.
        tile_size (int, optional): Size of the tiles used for processing the video. If None, the video is processed as a whole. Default is None.
        tile_overlap (int, optional): Overlap between tiles when the video is processed in parts. Default is None.
        restormer_task (str, optional): Specifies if a Restormer model task should be applied. Options are 'Disabled', 'Motion_Deblurring', 'Real_Denoising', 'Deraining'. Default is 'Disabled'.
        progress (Progress, optional): Gradio progress bar

    Returns:
        str: Path to the output video file after processing.
    """
    global container_constructor, container, hdsrnet_model, hat_model, restormer_model, should_continue
    should_continue = True
    # Check if input video is not None
    if filepath is None:
        return None
    # Check if input numeric values is Integer
    if isinstance(start_sec, float):
        start_sec = int(start_sec)
        gr.Info(f"Start sec change to integer value {start_sec}")
    if isinstance(end_sec, float):
        end_sec = int(end_sec)
        gr.Info(f"End sec change to integer value {end_sec}")
    if isinstance(out_fps, float):
        out_fps = int(out_fps)
        gr.Info(f"Out fps change to integer value {out_fps}")
    if isinstance(tile_size, float):
        tile_size = int(tile_size)
        gr.Info(f"Tile size change to integer value {tile_size}")
    if isinstance(tile_overlap, float):
        tile_overlap = int(tile_overlap)
        gr.Info(f"Tile overlap change to integer value {tile_overlap}")
    # Check input values validity
    if tile_size == 0:
        tile_size = None
    if tile_size and tile_size % 8 != 0:
        while tile_size % 8 != 0:
            tile_size += 1
        gr.Info(f"Tiling size changed to {tile_size}, because tile size must be multiple of 8")
    if out_fps == 0:
        out_fps = -1
        gr.Info(f"Output fps changed to original, because it can't be zero")
    if start_sec > end_sec:
        start_sec = 0
        gr.Info(f"Start_sec changed to {start_sec}, because it can't be greater than end_sec")
    if start_sec == end_sec:
        if start_sec != 0:
            start_sec -= 1
        else:
            end_sec += 1
        gr.Info(f"Start_sec and end_sec changed to {start_sec} and {end_sec}, because they can't be equal")
    if tile_size is not None and tile_size == tile_overlap:
        gr.Info(f"Tiling size changed to {tile_size + 8}, because it can't be equal to tile overlap")
        tile_size += 8  
    logger.debug(f"| start_sec: {start_sec} | end_sec: {end_sec} | resolution: {resolution} | out_fps: {out_fps} | upscaler: {upscaler} | tile_size: {tile_size} | tile_overlap: {tile_overlap} | restormer_task: {restormer_task}")
    # Init models
    if upscaler == 'HDSRNet':
        hdsrnet_model = init_hdsrnet()
    else:
        gr.Info(f"HAT model is selected. Upscale can take much longer! For faster performance use HDSRNet")
        hat_model = init_hat()
    if restormer_task != 'Disabled':
        restormer_model = init_restormer(restormer_task)
    frame_len = 1
    name = os.path.splitext(os.path.basename(filepath))[0]

    container = container_constructor(filepath, OUTPUT_PATH+name+'.mp4', width=-1, height=-1, out_fps=out_fps, 
                                        start_time=start_sec, duration=end_sec - start_sec)
    if out_fps == -1:
        out_fps = container.video_fps()
        
    for b in progress.tqdm(range(0, end_sec - start_sec, MAX_DURATION_BATCH), desc='🎠 Clip'):
        if not should_continue:
            break
        b_start = b + start_sec
        clip_duration = MAX_DURATION_BATCH if b_start+MAX_DURATION_BATCH<=end_sec else end_sec - b_start
        logger.debug("🎠 Clip {}s - {}s".format(b_start, b_start+clip_duration))
        for i in progress.tqdm(range(0, MAX_DURATION_BATCH if b_start+MAX_DURATION_BATCH<=end_sec else end_sec - b_start, frame_len), desc='🖼️ Processing step'):
            if b_start + i > container.total_duration or not should_continue:
                break
            logger.debug(f"🖼️ Processing step {i + 1}/{MAX_DURATION_BATCH if b_start+MAX_DURATION_BATCH<=end_sec else end_sec - b_start}...")
            video, audio = None, None
            try:
                video, audio = inference_step(vid=container, duration=frame_len, resolution=resolution, out_fps=out_fps, upscaler=upscaler, tile_size=tile_size, tile_overlap=tile_overlap, restormer_task=restormer_task)
            except torch.cuda.OutOfMemoryError:
                gr.Warning('Cuda out of memory on server. Please set appropriate Tile size and Tile overlap')
                return None
            except np.core._exceptions._ArrayMemoryError:
                gr.Warning('Ouf RAM. Try to change MAX_DURATION_BATCH parameter in shared.py module')
                return None
            if i == 0:
                video_all = np.zeros([out_fps*clip_duration, video.shape[1], video.shape[2], video.shape[3]], np.uint8)
                audio_all = audio
                
            video_all[i*out_fps: (i+1)*out_fps] = video
            
        logger.info(f"💾 Writing output video...")
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        container.remux(video_all, audio_all)
        del audio_all
        del video_all
    if not container.outSink:
        logger.info(f"Stop before inference step. Empty output")
        return None
    container.close()
    logger.info(f"✅ Done!")
    return OUTPUT_PATH+name+'.mp4'

def stop_inference():
    global should_continue
    should_continue = False
    gr.Info(f"Stopping... Please wait")

def on_video_change(filepath, start_sec, out_fps):
    """
    Processes a video file to extract and display its statistics, and updates sliders based on the video's duration.

    Args:
        filepath (str): The path to the video file.
        start_sec (int): The initial position of the playback slider, in seconds.
        out_fps (int): The output frames per second for the video processing.

    Returns:
        list: A list containing updated slider components and a string of video and audio statistics.
              The first slider is updated with the start position and maximum based on the video's duration.
              The second slider is updated to reflect the total duration of the video.
              The statistics string includes formatted information about the video and audio streams.
              None value for outputs_video, for update output video field
    """
    if filepath is None:
        stats = ""
        return start_sec, out_fps, stats, None, gr.Button(value="Stop", visible=True, variant='stop',interactive=False)
    global container
    logger.info(f"📹 Input video: {filepath.split('/')[-1]}")
    name = os.path.splitext(os.path.basename(filepath))[0]
    container = container_constructor(filepath, OUTPUT_PATH+name+'.mp4', width=-1, height=-1, out_fps=out_fps, start_time=0, duration=1)
    max_duration = container.get_total_duration()
    container.get_stats()
    # output_video = gr.Video(interactive=False, height=container.get_stats()['video']['height']*10)
    stats = [ (key, value) for key,value in container.get_stats()['video'].items()]
    stats = ""
    stats += '**video:**  \n    '
    for key,value in container.get_stats()['video'].items():
        stats += "   {}: {}".format(key, value)
    stats += '  \n  **audio:**  \n    '
    for key,value in container.get_stats()['audio'].items():
        stats += " {}:{}".format(key, value)
    del container
    outputs_video = None
    return [gr.Slider(value=start_sec if start_sec<max_duration else 0, maximum=max_duration), 
            gr.Slider(value=round(max_duration) , maximum=max_duration),\
            stats, outputs_video, gr.Button(value="Stop", visible=True, variant='stop',interactive=True)
            ]

def main_tab():
    """
    Constructs the main tab interface for video processing in a web application using Gradio. 
    This interface allows users to upload a video, set processing parameters, and submit the video for processing.

    Returns:
        None: This function sets up the UI components but does not return any values.
    """
    with gr.Row(equal_height=True):
        with gr.Column():
            inputs_video = gr.Video(sources="upload", interactive=True)
            stats = gr.Markdown()

            with gr.Group() as meta:
                start_sec = gr.Slider(minimum=0, maximum=3000, step=1, value=0, label="start_sec", interactive=True)
                end_sec = gr.Slider(minimum=1, maximum=3000, step=1, value=6, label="end_sec", interactive=True)
            with gr.Group() as resolution:
                resolution = gr.Radio(choices=['x4','360p', '480p', '720p', '1080p', '2K', '4K'], label="Resolution",value='x4', interactive=True)
            with gr.Group() as video_setup:
                out_fps = gr.Slider(minimum=-1, maximum=60, step=1, value=-1, label="output_fps (original fps: -1)", interactive=True)
            with gr.Group() as upscaler_setup:
                upscaler = gr.Radio(choices=['HDSRNet','HAT'], label="Upscaler",value='HDSRNet', interactive=True)
            with gr.Group() as tiling_setup:
                tile_size = gr.Slider(minimum=0, maximum=240, value=208, step=8, label="Tile Size", visible=True)
                tile_overlap = gr.Slider(minimum=0, maximum=32, value=16, step=8, label="Tile Overlap", visible=True)
            with gr.Row():
                restormer_task = gr.Dropdown(value='Disabled', choices=['Disabled','Motion Deblurring','Real Denoising', 'Deraining'], label='Restormer', interactive=True)
            with gr.Group() as control:
                with gr.Row():
                    submit = gr.Button("Submit")
            stop_button = gr.Button(value="Stop", visible=True, variant='stop',interactive=False)
        with gr.Column():
            outputs_video = gr.Video(interactive=False)
    examples = gr.Examples([
        ['./assets/examples/video_example_1_donut_144p.mp4'],
        ['./assets/examples/video_example_2_photo_240p.mp4']
    ], inputs=[inputs_video, start_sec, end_sec], outputs=[inputs_video])
    submit.click(fn=predict_fn, inputs=[inputs_video, start_sec, end_sec, resolution, out_fps, upscaler, tile_size, tile_overlap, restormer_task], outputs=[outputs_video])
    inputs_video.change(fn=on_video_change, inputs=[inputs_video, start_sec, out_fps], outputs=[start_sec, end_sec, stats, outputs_video, stop_button])
    stop_button.click(fn=stop_inference)

with gr.Blocks(title="Video Super Resolution") as app:
    gr.Label("Video Super Resolution")
    with gr.Tab(label="Super Resolution"):
        main_tab()


if __name__ == '__main__':
    logger.info('🔼 Start Super Resolution web app')
    app.queue().launch(share=True)
    logger.info('🔽 Stop Super Resolution web app')