IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108

BATCH_SIZE = 32
NUM_CLASSES = 27
LEARNING_RATE = 0.001
EPOCHS = 100
VALIDATION_STEPS = 32

SINGLE_FRAME_ENCODER_DIMS = 1024  # output dimension of single frame encoder
NUM_FRAMES = 10  # number of frames
MIN_FPS = 10  # recommended to be less than or equal to 12 because the dataset has a max fps of 12

TRAIN_BRANCHES = 10  # number of branches the training model should have (in addition to  the main branch)
