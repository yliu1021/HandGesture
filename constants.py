IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108

TRAINING_RUN = 'run7'
BATCH_SIZE = 32
NUM_CLASSES = 27
LEARNING_RATE = 1*10**0
EPOCHS = 35
VALIDATION_STEPS = 32
PRUNING_START_EPOCH = 25
PRUNING_END_EPOCH = 35
PRUNE_FREQ = 2

SINGLE_FRAME_ENCODER_DIMS = 512  # output dimension of single frame encoder
NUM_FRAMES = 20  # number of frames
MIN_FPS = 10  # recommended to be less than or equal to 12 because the dataset has a max fps of 12
