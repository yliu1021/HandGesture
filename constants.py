IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108

TRAINING_RUN = 'run23'
BATCH_SIZE = 64
NUM_CLASSES = 27
LEARNING_RATE = 1*10**-3
EPOCHS = 100
VALIDATION_STEPS = 200
PRUNING_START_EPOCH = 25
PRUNING_END_EPOCH = 75
PRUNE_FREQ = 5

NUM_FRAMES = 12     # number of frames
FAST_FRAMES = 16    # number of frames with high FPS
MIN_FPS = 1         # recommended to be less than or equal to 12 because the dataset has a max fps of 12
