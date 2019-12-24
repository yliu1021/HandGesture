IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108

TRAINING_RUN = 'run1'
BATCH_SIZE = 32
NUM_CLASSES = 27
LEARNING_RATE = 1*10**-3
EPOCHS = 100
VALIDATION_STEPS = 32

SINGLE_FRAME_ENCODER_DIMS = 512  # output dimension of single frame encoder
NUM_FRAMES = 20  # number of frames
MIN_FPS = 10  # recommended to be less than or equal to 12 because the dataset has a max fps of 12
