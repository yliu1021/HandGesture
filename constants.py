IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108

TRAINING_RUN = 'run14'
BATCH_SIZE = 64
NUM_CLASSES = 27
LEARNING_RATE = 1*10**-1
EPOCHS = 100
VALIDATION_STEPS = 200
PRUNING_START_EPOCH = 25
PRUNING_END_EPOCH = 75
PRUNE_FREQ = 5

NUM_FRAMES = 8  # number of frames
MIN_FPS = 3     # recommended to be less than or equal to 12 because the dataset has a max fps of 12
