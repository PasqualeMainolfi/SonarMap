from smap.utils import FFTParams, KernelParams, KernelType, CostellationParams
import os
from pathlib import Path

# [MODEL_TO_USE]
ROOT = Path(__file__).resolve().parent.parent
PVQ_MODEL_PATH = str(ROOT / "models/vq/pvq_best_model.pth")

# [PATHS]
CB_AUDIO_DATASET_PATH     = "/Volumes/T7/fma-dataset/fma_small"
DATASETS_ROOT_DESTINATION = "/Volumes/T7"

# [FFT]
ETA         = 1e-12
SR          = 22050
FFT_WINSIZE = 2048
FFT_HOPSIZE = FFT_WINSIZE // 2
FFT_WINDOW  = "hann"
FFT_HOPSEC  = FFT_HOPSIZE / SR
FFT_SRHOP   = SR / FFT_HOPSIZE
FFT_PARAMS  = FFTParams(wsize=FFT_WINSIZE, hsize=FFT_HOPSIZE, w=FFT_WINDOW, sr=SR)

# [COSTELLATION]
KERNEL_SHAPE        = (7, 5)
KERNEL_HOP          = (3, 2)
KERNEL_THRESHOLD    = -75.0
N_PAIRS             = 7
EXTRA_PAIRS         = N_PAIRS + 3
DT_MAX              = 1.0
KERNEL_TYPE         = KernelType.MAXIMUM_FILTER
KERNEL_PARAMS       = KernelParams(ktype=KERNEL_TYPE, kshape=KERNEL_SHAPE, khop=KERNEL_HOP, kthreshold=KERNEL_THRESHOLD)
COSTELLATION_PARAMS = CostellationParams(n_pairs=N_PAIRS, dt_max=DT_MAX)

PHASE_PATCH_SHAPE = (6, 2)
PATCH_H            = 2 * PHASE_PATCH_SHAPE[0] + 1
PATCH_W            = 2 * PHASE_PATCH_SHAPE[1] + 1
PATCH_DIM          = 2 * PATCH_H * PATCH_W

# [SCALAR BIN TOKENIZATION]
# Every scalar quantity is tokenized via deterministic uniform bins — no learned codebook.
# Step sizes chosen so residual quantization error is perceptually insignificant.
DB_MIN             = -80.0
DB_MAX             = 5.0
DB_STEP            = 0.5
DB_DICT_SIZE       = int(round((DB_MAX - DB_MIN) / DB_STEP)) + 1    # 171

DDB_MIN            = -60.0
DDB_MAX            = 60.0
DDB_STEP           = 0.5
DDB_DICT_SIZE      = int(round((DDB_MAX - DDB_MIN) / DDB_STEP)) + 1  # 241

PHASE_DICT_SIZE    = 256                                             # shared anchor/target delta-phase codebook

DT_STEP            = FFT_HOPSEC
DT_DICT_SIZE       = int(round(DT_MAX / DT_STEP)) + 1                # ~22 at sr=22050 / hop=1024

# [CODEBOOKS — patch only]
PATCH_DICT_SIZE    = 1024                              # shared anchor+target patch codebook (per RVQ stage)
PVQ_N_STAGES       = 24                                # residual VQ stages
FBIN_DICT_SIZE     = FFT_WINSIZE // 2 + 1              # 1025 — exact freq bins
DFBIN_SHIFT        = FFT_WINSIZE // 2                  # signed shift [-1024, 1024] → [0, 2048]
DFBIN_DICT_SIZE    = 2 * DFBIN_SHIFT + 1               # 2049

MAX_TIME_INTERVAL  = 10  # sec
MAX_TIME_FRAMES    = int(MAX_TIME_INTERVAL / FFT_HOPSEC)
TIME_DICT_SIZE     = MAX_TIME_FRAMES

# [STRUCTURAL]
PAD_ID             = 0
BOS_ID             = 1
EOS_ID             = 2
C_START_ID         = 3
C_END_ID           = 4

# Grammar (deterministic bin tokens, no scalar VQ):
# C_START → db → aphase → fbin → patch(×24) → [ddb → tphase → dt → dfbin → patch(×24)]* → C_END → TIME_SHIFT
DB_OFFSET          = 5
APHASE_OFFSET      = DB_OFFSET + DB_DICT_SIZE
FBIN_OFFSET        = APHASE_OFFSET + PHASE_DICT_SIZE
DDB_OFFSET         = FBIN_OFFSET + FBIN_DICT_SIZE
TPHASE_OFFSET      = DDB_OFFSET + DDB_DICT_SIZE
DT_OFFSET          = TPHASE_OFFSET + PHASE_DICT_SIZE
DFBIN_OFFSET       = DT_OFFSET + DT_DICT_SIZE
PATCH_OFFSET       = DFBIN_OFFSET + DFBIN_DICT_SIZE
TIME_SHIFT_OFFSET  = PATCH_OFFSET + PATCH_DICT_SIZE
DICT_SIZE          = TIME_SHIFT_OFFSET + TIME_DICT_SIZE

# RQ-Transformer outer vocab retained for interop with paused smap-transformer/:
PATCH_BLOCK_ID     = DICT_SIZE
OUTER_DICT_SIZE    = DICT_SIZE + 1
STRUCTURAL_TOKENS  = [PAD_ID, BOS_ID, EOS_ID, C_START_ID, C_END_ID, PATCH_BLOCK_ID]

MAX_WORKER_CPU     = max(4, os.cpu_count() or 1)

# [PATCH VQ]
PVQ_BATCH_SIZE      = 512
PVQ_MAX_BATCH_EPOCH = 1000
PVQ_LR              = 1e-4
PVQ_MIN_LR          = 5e-6
PVQ_LATENT_DIM      = 128
PVQ_EPOCHS          = 30
