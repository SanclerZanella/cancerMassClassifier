import os
import sys
import pkgutil

# Add the project root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.documentation import templates


class DocumentationConstants:
    def __init__(self):
        self.PROJECT_TITLE = "Cancer Mass Classifier ML Project"
        self.FILE_NAME = "README.md"
        self.TEMPLATE_FOLDER = pkgutil.find_loader(templates.__name__)
        self.TEMPLATE_FILE_NAME = "README.txt"
