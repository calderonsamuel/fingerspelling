"""
pytest configuration and fixtures for the fingerspelling project.
"""

import sys
from pathlib import Path

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
