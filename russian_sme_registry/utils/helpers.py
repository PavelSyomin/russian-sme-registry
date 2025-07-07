import subprocess
import sys
from typing import Tuple

MIN_JAVA_VERSION = (8, 0)


def get_java_version() -> Tuple[int, int]:
    """Get the installed Java version.

    Returns:
        Tuple[int, int]: Major and minor version numbers of Java.
        Returns (0, 0) if Java is not installed or version cannot be determined.
    """
    try:
        result = subprocess.run(['java', '-version'], capture_output=True, text=True)
        # Java version info is written to stderr
        version_str = result.stderr.split('\n')[0]
        # Extract version numbers from string like "java version "1.8.0_291""
        version = version_str.split('"')[1].split('.')
        return (int(version[0]), int(version[1]))
    except (subprocess.SubprocessError, FileNotFoundError, IndexError, ValueError):
        return (0, 0)


def check_java_installation(min_major: int = 8, min_minor: int = 0) -> bool:
    """Check if Java is installed and meets minimum version requirements.

    Args:
        min_major: Minimum required major version
        min_minor: Minimum required minor version

    Returns:
        bool: True if Java is installed and meets version requirements, False otherwise
    """
    major, minor = get_java_version()
    if major == 0 and minor == 0:
        return False
    return major > min_major or (major == min_major and minor >= min_minor)


def require_java(min_major: int = MIN_JAVA_VERSION[0], min_minor: int = MIN_JAVA_VERSION[1]):
    """Check if Java is installed and meets minimum version requirements. If not, print a warning and exit.

    Args:
        min_major: Minimum required major version
        min_minor: Minimum required minor version

    This function should be called before any Spark operations.
    """
    if not check_java_installation(min_major, min_minor):
        major, minor = get_java_version()
        if major == 0 and minor == 0:
            print("Error: Java is not installed or not accessible.")
        else:
            print(f"Error: Installed Java version {major}.{minor} is too old.")
        print(f"This application requires Java {min_major}.{min_minor} or newer to run PySpark operations.")
        print("Please install a compatible version of Java and try again.")
        sys.exit(1)


def print_config(config: dict):
    print("Current configuration:")
    for key, value in config.items():
        key = key.replace("_", "-")
        if isinstance(value, list):
            value = ", ".join(value)
        print(f"- {key}: {value}")
