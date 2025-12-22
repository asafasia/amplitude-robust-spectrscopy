"""Configuration settings for the echospec package."""

from pathlib import Path

# Default data directory for saving experiment results
# Uses absolute path relative to user's home directory
DATA_PATH = Path.home() / "Developer" / "amplitude-robust-spectroscopy" / \
    "data" / "experiments"

# You can override this path by setting it to your preferred location
# Example: DATA_PATH = Path.home() / "my_experiments" / "data"


# Check if data path exists
def check_data_path() -> bool:
    """
    Check if the configured DATA_PATH exists.

    Returns
    -------
    bool
        True if path exists, False otherwise

    Raises
    ------
    FileNotFoundError
        If DATA_PATH does not exist
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data path does not exist: {DATA_PATH}\n"
            f"Please create the directory or update DATA_PATH in echospec/utils/config.py"
        )
    return True


def ensure_data_path() -> Path:
    """
    Ensure the DATA_PATH exists by creating it if necessary.

    Returns
    -------
    Path
        The DATA_PATH after ensuring it exists
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    return DATA_PATH


# Perform initial check when module is imported
_path_exists = check_data_path()
