# Imports
from pathlib import Path

# Paths
project_path = Path("/rds/project/rds-8b3VcZwY7rY/projects/dated_selection")
data_dir = project_path / "data"

slim_binary_path = "slim"
slim_pytest_script = "./slim/pytest.slim"

include: "snakefiles/basic.snk"

