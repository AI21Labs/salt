from pathlib import Path

TEXT = "text"
VECTOR = "vector"
DATE = "dt"
LABEL = "label"
PRED = "pred"
PROB = "prob"
CLUSTER = "cluster"
MEAN_DISTANCE = "mean_distance"

LABELS_SEP = ","

NA = "<NA>"
SKIP = "Skip"
ALL = "All"

PROJECT_STATE_KEY = "project"
EDITED_DF_KEY = "edited_df"

PROJECTS_DIR = str(Path.home().joinpath("SaltProjects"))
EMBEDDINGS_FILE_NAME = "embeddings.pkl"
DUMP_INTERVAL = 10
