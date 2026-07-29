"""
Global configuration for the MetVis FROSI release package.
"""
import os
import platform as _platform

# ============================================================
# Paths
# ============================================================

DATA_ROOT = "dataset"
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
CACHE_DIR = "./cache"
AUX_CACHE_ROOT = "cache/aux_maps"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AUX_CACHE_ROOT, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "baseline_results"), exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "ablation_results"), exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "summary"), exist_ok=True)

# ============================================================
# NOAA meteorological data (Stage-1 pretrain / Stage-3 aligner)
# Layout: dataset/NOAA/{2017..2024}/*.csv
# ============================================================
NOAA_ROOT = os.path.join(DATA_ROOT, "NOAA")

# ============================================================
# FROSI synthetic fog dataset
# Layout: dataset/FROSI/Fog/{50,100,150,200,250,300,400}/*.png
# Visibility label = Fog subdirectory name (meters).
# ============================================================
FROSI_ROOT = os.path.join(DATA_ROOT, "FROSI")
FROSI_FOG_DIR = os.path.join(FROSI_ROOT, "Fog")
FROSI_MSK_DIR = os.path.join(FROSI_ROOT, "Mask")
FROSI_VIS_DIRS = [50, 100, 150, 200, 250, 300, 400]

# ============================================================
# Visibility classes (meters)
# ============================================================
MAX_VIS = 50000.0
USE_METER_LABELS = True
VIS_BINS = [0, 50, 100, 200, 500, 1000, float("inf")]
VIS_CLASS_NAMES = ["0", "1", "2", "3", "4", "5"]
NUM_VIS_CLASSES = len(VIS_CLASS_NAMES)
VIS_MAE_LOW_CLASSES = [0, 1, 2, 3]
VIS_MAE_HIGH_CLASSES = [4, 5]

# ============================================================
# Image
# ============================================================
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# ============================================================
# Meteorological feature columns
# ============================================================
NOAA_FEATURES = [
    "VIS_DISTANCE", "TEMPERATURE", "DEW_POINT",
    "WIND_DIRECTION", "WIND_SPEED", "CLOUD_HEIGHT", "SEA_LEVEL_PRESSURE",
]
NUM_METEO_FEATURES = len(NOAA_FEATURES) - 1

# ============================================================
# Temporal
# ============================================================
TIME_WINDOW_HOURS = 12
SEQ_LEN = 24
PRED_LEN = 1
NOAA_METEO_CACHE = os.path.join(CACHE_DIR, "noaa_stations")

# ============================================================
# Model dims
# ============================================================
CNN_BACKBONE = "resnet50"
IMAGE_FEAT_DIM = 512
TIME_FEAT_DIM = 64
FUSED_DIM = 256
TIMESNET_D_MODEL = 64
TIMESNET_D_FF = 128
TIMESNET_TOP_K = 3
TIMESNET_N_KERNELS = 6
TIMESNET_NUM_LAYERS = 2

# ============================================================
# Training
# ============================================================
BATCH_SIZE = 32
NUM_WORKERS = 1
PIN_MEMORY = True
MAIN_EPOCHS = 50
NOAA_PRETRAIN_EPOCHS = 30
MAIN_LR = 5e-5
NOAA_LR = 1e-3
NOAA_WEIGHT_DECAY = 1e-4
MAIN_WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0

# FROSI split 7:1:2
FROSI_TRAIN_RATIO = 0.7
FROSI_VAL_RATIO = 0.1
FROSI_SPLIT_SEED = 42

# Kept for compatibility with shared training helpers (unused for FROSI stratified split)
TEMPORAL_TRAIN_RATIO = 0.8
TEMPORAL_VAL_RATIO = 0.1
IMAGE_TRAIN_RATIO = 0.70
IMAGE_VAL_RATIO = 0.15
IMAGE_TEST_RATIO = 0.15
IMAGE_SPLIT_SEED = 42
IMAGE_SAMPLER_MODE = "balanced"
IMAGE_MILD_POWER = 0.5

STAGE2_BEST_MAE_TIE_TOL = 50.0
STAGE3_BEST_MAE_TIE_TOL = 50.0
MAIN_FREEZE_FUSION = 5
MAIN_STAGE3_EPOCHS = 15
MAIN_DUAL_FREEZE = True
BASELINE_BEST_MAE_TIE_TOL = 50.0
STAGE3_DUAL_FREEZE = False
STAGE3_DUAL_FUSION_DISTILL_EPOCHS = 0
STAGE3_UGG_RAMP_EPOCHS = 5
STAGE3_DUAL_ALPHA_RAMP_EPOCHS = 15
STAGE3_DUAL_TRANSITION_ALPHA = 0.15
STAGE3_DEFAULT_TRANSITION_ALPHA = 0.15
STAGE3_GATE_ANCHOR_WEIGHT = 0.1
STAGE3_BLEND_ANCHOR_WEIGHT = 0.8
STAGE3_BLEND_REG_ANCHOR_WEIGHT = 15.0
STAGE3_BLEND_REG_USE_RAMP = False
STAGE3_SUP_REG_WEIGHT_RAMP = 5.0
STAGE3_SUP_REG_SMOOTH_WEIGHT = 0.1
STAGE3_FUSE_REG_USE_EDL = True
STAGE3_FUSE_REG_EDL_USE_NIG = False
STAGE3_FUSE_REG_EDL_WEIGHT = 0.5
STAGE3_GAMMA_DISTILL_WEIGHT = 2.0
STAGE3_LOW_VIS_REG_WEIGHT = 3.0
STAGE3_LOW_VIS_MAX_CLS = 3
STAGE3_CONTRA_WEIGHT_RAMP = 0.05
STAGE3_DUAL_DISTILL_MAX_EPOCH = 1
STAGE3_DEFAULT_DISTILL_MAX_EPOCH = 1
STAGE3_MIN_TOTAL_FLOOR = 1e-3
STAGE3_GATE_PREWARM_WEIGHT = 0.05
STAGE3_LABEL_SMOOTH = 0.05
STAGE3_TRAIN_MIN_ALPHA = 0.03
STAGE3_CONTRA_MAX = 10.0
STAGE3_TS_PROJ_INIT_GAIN = 0.01

# ============================================================
# Loss weights
# ============================================================
LOSS_CLS_WEIGHT = 0.1
LOSS_REG_WEIGHT = 1.0
LOSS_TS_WEIGHT = 0.8
LOSS_CONTRA_WEIGHT = 0.2
LOSS_PRED_WEIGHT = 0.3
STAGE3_SUP_CLS_WEIGHT = 1.0
STAGE3_SUP_REG_WEIGHT = 5.0
STAGE3_FUSE_EDL_SCALE = 0.1
CONTRA_TEMP = 0.07

EDL_ANNEAL_START = 0.01
EDL_ANNEAL_STEP = 0.025
EDL_REG_COEFF = 1e-4

KD_TEMP = 2.0
KD_WEIGHT_MAX = 0.4
KD_WARMUP_EPOCHS = 15
KD_DIRECTION = "ts2img"
USE_KD = False

UGG_EPISTEMIC_CLIP = 5.0
USE_UNC_GATE = True

TCAM_INPUT_DIM = 20
TCAM_HIDDEN_DIM = 64
TCAM_FEAT_DIM = 512
TCAM_NUM_LAYERS = 2
TCAM_MAX_DELTA_H = 24.0
USE_TCAM = True
TCAM_DEFAULT_DELTA_H = 0.0
TCAM_PRED_WEIGHT = 0.3
TCAM_DELTA_PERIODS = [1.0, 2.0, 6.0, 12.0, 24.0]

DEFAULT_UGG_MASK = {"u_img": True, "u_ts": True, "avail": True, "feat_sim": True}
UGG_UNC_MODE_DEFAULT = "default"
STAGE3_DELTA_INJECT_PROB = 0.5
STAGE3_DELTA_MIN_H = 0.5
STAGE3_DELTA_MAX_H = 6.0

TIME_PRIOR_WEIGHT = 0.1
TIME_PRIOR_TEMP = 0.2
TIME_ALIGN_CYCLIC = True
TIME_ALIGN_YEAR_SECONDS = 365 * 24 * 3600
TIME_ALIGN_MAX_GAP_HOURS = 24 * 365
TIME_ALIGN_SIGMA_SECONDS = 30 * 24 * 3600
TIME_ALIGN_TIME_ONLY_FALLBACK = True
TIME_ALIGN_CACHE_ENABLE = True
TIME_ALIGN_CACHE_DIR = os.path.join(CACHE_DIR, "time_align")
TIME_ALIGN_CACHE_MAX_ITEMS = 5000
TIME_ALIGN_CACHE_BUCKET_MINUTES = 60
TIME_ALIGN_CACHE_DISK = True
TIME_ALIGN_DEBUG = False
TIME_ALIGN_DEBUG_MAX_SAMPLES = 2000
TIME_ALIGN_RANDOM_STATIONS = 10

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = None
LABEL_SMOOTHING = 0.05

# ============================================================
# Checkpoints / inference
# ============================================================
INFER_ALPHA_MIN = 0.0
INFER_ALPHA_MAX = 0.6
INFER_WINDOW_FULL = 24
INFER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
NOAA_PRETRAIN_CKPT = os.path.join(CHECKPOINT_DIR, "noaa_pretrain.pth")
STAGE2_CKPT = os.path.join(CHECKPOINT_DIR, "stage2_image_best.pth")
STAGE3_CKPT = os.path.join(CHECKPOINT_DIR, "stage3_joint_best.pth")


def stage2_ckpt_path(dataset_name: str, suffix: str = None) -> str:
    """Stage-2 image-branch checkpoint path (per dataset)."""
    if suffix:
        safe = str(suffix).replace("/", "_").replace("\\", "_")
        return os.path.join(CHECKPOINT_DIR, f"stage2_image_{dataset_name}_{safe}.pth")
    return os.path.join(CHECKPOINT_DIR, f"stage2_image_{dataset_name}.pth")


SEED = 42
UNC_VIS_DIR = os.path.join(LOG_DIR, "uncertainty_vis")
SAMPLER_MODE = "balanced"
CLASS_WEIGHT_MODE = "sqrt_inv"
BALANCED_SAMPLES_PER_CLASS = 2000
MAX_CLASS5_RATIO = 0.5
TIME_KERNEL_SIGMA = 3600
