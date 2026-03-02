from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "paper" / "paper_figures"

SEED = 42

CONGRESSES = list(range(100, 119))
TRAIN_CONGRESSES = list(range(104, 114))
VAL_CONGRESS = 114
TEST_CONGRESSES = [115, 116, 117]
ANALYSIS_CONGRESS = 118

THRESHOLD_TAU = 0.5
MIN_VOTES = 50
MIN_SHARED_VOTES = 20
DEFECTION_THRESHOLD = 0.10

HIDDEN_DIM = 32
N_HEADS = 4
N_TEMPORAL_HEADS = 2
DROPOUT = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.5
GRAD_CLIP = 1.0
BATCH_SIZE = 32
COALITION_PAIRS = 200

DEM_COLOR = "#2166ac"
REP_COLOR = "#b2182b"
CROSS_COLOR = "#d95f02"

MCCARTHY_HOLDOUTS = [
    "BIGGS", "BISHOP", "BOEBERT", "BRECHEEN", "BUCK",
    "CLOUD", "CLYDE", "CRANE", "DONALDS", "GAETZ",
    "GOOD", "GOSAR", "HARRIS", "LUNA", "MILLS",
    "NORMAN", "OGLES", "PERRY", "ROSENDALE", "ROY",
]

VOTEVIEW_URLS = {
    "HSall_members.csv": "https://voteview.com/static/data/out/members/HSall_members.csv",
    "HSall_votes.csv": "https://voteview.com/static/data/out/votes/HSall_votes.csv",
    "HSall_rollcalls.csv": "https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv",
}
