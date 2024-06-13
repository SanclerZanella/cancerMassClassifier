import os
import sys
from jinja2 import Environment, FileSystemLoader, Template
from main import config

# Add the project root directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.data_processing.diagnosedatasetfetcher import DiagnoseDatasetFetcher
from main.constants.constants import DocumentationConstants

class ReadmeGenerator:
    def __init__(self):
        doc_constants = DocumentationConstants()
        dataset = DiagnoseDatasetFetcher()

        self.project_title = doc_constants.PROJECT_TITLE
        self.dataset_characteristics = dataset.dataset_description()
        self.template_folder = "\documentation\\templates"
        self.template_file_name = doc_constants.TEMPLATE_FILE_NAME
        self.file_name = "README.md"
        self.base_path = str(config.BASE_DIR)

    def __load_template(self) -> Template:
        file_loader = FileSystemLoader(str(config.APP_DIR) + self.template_folder)
        env = Environment(loader=file_loader)

        return env.get_template(self.template_file_name)

    def render_template(self) -> str:
        context = {
            "PROJECT_TITLE": self.project_title,
            "DATASET_CHARACTERISTICS": self.dataset_characteristics
        }
        template = self.__load_template()

        return template.render(context)

    def generate_readme(self):
        file_name = self.file_name
        with open(self.base_path + f"\{file_name}", "w") as file:
            file.write(self.render_template())


def generate():
    render = ReadmeGenerator()
    render.generate_readme()


if __name__ == "__main__":
    generate()
