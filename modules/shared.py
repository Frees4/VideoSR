RESTORMER_BASE_MODELS_PATH = 'models/Restormer'
RESTORMER_WEIGHTS = {'Motion Deblurring': 'models/Restormer/motion_deblurring.pth',
                     'Real Denoising': 'models/Restormer/real_denoising.pth',
                     'Deraining': 'models/Restormer/deraining.pth'}
RESTORMER_PRETRAINED_PATHS = {'Motion Deblurring': 'https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth',
                              'Real Denoising': 'https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth',
                              'Deraining': 'https://github.com/swz30/Restormer/releases/download/v1.0/deraining.pth'}
HAT_ROOT_PATH = 'lib/HAT'
HAT_OPTIONS_PATH = 'lib/HAT/options/test/HAT_SRx4_ImageNet-pretrain.yml'
OUTPUT_PATH = './outputs/'
LOG_DIR = 'logs/'
LOG_FILE = 'log.log'
RESOLUTIONS = {
    '360p': (640, 360),
    '480p': (848, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '2K': (2560, 1440),
    '4K': (3840, 2160)
}
MAX_DURATION_BATCH = 30
SCALE = 4