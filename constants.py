IMAGE_WIDTH = 192
IMAGE_HEIGHT = 108

TRAINING_RUN = 'run8'
BATCH_SIZE = 32
NUM_CLASSES = 27
LEARNING_RATE = 1*10**-2
EPOCHS = 100
VALIDATION_STEPS = 50
PRUNING_START_EPOCH = 25
PRUNING_END_EPOCH = 75
PRUNE_FREQ = 5

NUM_FRAMES = 20  # number of frames
MIN_FPS = 8  # recommended to be less than or equal to 12 because the dataset has a max fps of 12
