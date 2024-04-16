# Imports
from pathlib import Path
import pandas as pd
import tskit
import numpy as np

from src import extract_tree_stats

# Paths
project_path = Path("/rds/project/rds-8b3VcZwY7rY/projects/dated_selection")
data_dir = project_path / "data"

slim_binary_path = "slim"
slim_pytest_script = "./slim/pytest.slim"

include: "snakefiles/basic.snk"
include: "snakefiles/paper_simulations.snk"

