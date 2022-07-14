import pathlib

MATAN_API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjanRzN2M2YjA1MmI5MDg3OGYwMmtjMjg0Iiwib3JnYW5pemF0aW9uSWQiOiJjanRzN2M2YWw1aHMzMDc5OXFnOTkwMzl2IiwiYXBpS2V5SWQiOiJjbDR3ZTUycXoydDhxMDd4bTNmbm1iY3lyIiwic2VjcmV0IjoiZWI2ZmRhNGQ4NWNmYmU2MjAzYmRkYmM5OTYzZjVkNGYiLCJpYXQiOjE2NTYzMTMyNDQsImV4cCI6MjI4NzQ2NTI0NH0.W7szDFavgI2DZscmCMe3UR1k5fKX4NYXlhvupDM_mkg"
# AMIT_API = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDNpcW4wdHExdjh3MDc5MjlhdHJnd21xIiwib3JnYW5pemF0aW9uSWQiOiJjbDNpcW4wdGUxdjh2MDc5MjZvbDBidjdnIiwiYXBpS2V5SWQiOiJjbDRvNDB0enEyODl5MDdiODczenAzZm1iIiwic2VjcmV0IjoiM2RiNTE1MDQ0ZGIyMWZiN2M5YTg5N2NmYjA3YzM5MDYiLCJpYXQiOjE2NTU4MTI1MjAsImV4cCI6MjI4Njk2NDUyMH0.T3SpiCBabIP1rSu1yJHeA9yq77FOtB-yXl0eU2WLtuw"
LB_API_KEY = MATAN_API
# ENDPOINT = "https://api.labelbox.com/graphql"


### Uncomment the object detection example to use with object detection projects


MATAN_PROJECT_ID = 'cl06bz2uhlkyj0zbhdn4ygg5p' #Matan
AMIT_PROJECT_ID = 'cl583jbmzk7d5070e358e8np7'
PROJECT_ID = MATAN_PROJECT_ID

DATASETS = ['cl0i77zs2gwhz0zbk6o7wee7g', 'cl0i6fncubzve10aldsnrclfg', 'cl0i67hplgnve0zbk1oydhbj5', 'cl09kttll4ukm0zae3984aykf', 'cl09kn3e05qhz0z79cfk44obv', 'cl09jsep05ia210892ecy1uu3'] #Matan labelbox dataset ids attached to the project
# DATASETS = ['cl09kttll4ukm0zae3984aykf'] #Matan labelbox dataset ids attached to the project

BASE_PATH = pathlib.Path('/media/UbuntuData3/Users_Data/amitp/PycharmProjects/mal-coral')
GOOGLE_DRIVE_ID = '1TgHVJxOL70Cn7hMfe69xfJitmdQj2vsE' #need to enter the folder drive id

# Object detection example
# MODE = 'object-detection'
# DATA_LOCATION = pathlib.Path('obj-data')

# Segmentation example
DATA_LOCATION = pathlib.Path('Matan/seg-data')
MODE = 'segmentation-rle'

## Universal configuration
DOWNLOAD_IMAGES = False  # Download data from labelbox. Set false for re-runs when data already exists locally
VALIDATION_RATIO = 0.2  # Validation data / training data ratio
NUM_CPU_THREADS = 8  # for multiprocess downloads  # TODO Naama was 8
NUM_SAMPLE_LABELS = 0  # Use 0 to use all of the labeled training data from project. Otherwise specify number of labeled images to use. Use smaller number for faster iteration.
PRELABELING_THRESHOLD = 0.6  # minimum model inference confidence threshold to be uploaded to labelbox
HEADLESS_MODE = False  # Set True to skip previewing data or model results

DETECTRON_DATASET_TRAINING_NAME = 'prelabeling-train'
DETECTRON_DATASET_VALIDATION_NAME = 'prelabeling-val'

EXISTING_JSON_TRAINING_PATH = 'traindataset_dict.pickle'
# EXISTING_JSON_TRAINING_PATH = None
EXISTING_JSON_VAL_PATH      = 'valdataset_dict.pickle'
# EXISTING_JSON_VAL_PATH      = None

PROPOSAL_GENERATOR_NAME = "CoralRPN" #"RPNOutputs"
ROI_HEADS_NAME = "CoralStandardROIHeads" #"StandardROIHeads"

IS_TRAINING = False

# configurations for data utils that I took out
train = 'train'
val = 'val'
inference = 'inference'
masks = 'masks'
tmp = 'tmp'

# configuraions for training that I took out
EVAL_PERIOD = 150
DATALOADER_NUM_WORKERS = 0
IMS_PER_BATCH = 2
BASE_LR = 0.00125
MAX_ITER = 1000
BATCH_SIZE_PER_IMAGE = 256
