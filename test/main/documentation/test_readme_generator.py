import os
import unittest
from main.documentation.readme_generator import ReadmeGenerator
from test.main.utils import README_TEST


class ReadmeGeneratorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This runs once before all tests
        cls.target_class = ReadmeGenerator()

    @classmethod
    def tearDownClass(cls):
        # This runs once after all tests
        del cls.target_class

    def test_render_template(self):
        class_to_be_tested = self.target_class
        class_to_be_tested.project_title = "Test README"
        class_to_be_tested.dataset_characteristics = "TEST DATA SET CHARACTERISTICS"

        template_renderer = class_to_be_tested.render_template()

        # Check template content type
        self.assertEqual(type(template_renderer), str)

        # Check template content
        self.assertEqual(template_renderer.strip(), README_TEST)

    def test_generate_readme(self):
        class_to_be_tested = self.target_class
        class_to_be_tested.file_name = "README_TEST.md"
        class_to_be_tested.project_title = "Test README"
        class_to_be_tested.dataset_characteristics = "TEST DATA SET CHARACTERISTICS"
        class_to_be_tested.base_path = os.getcwd()

        class_to_be_tested.generate_readme()
        expected_file_path = os.getcwd() + "\README_TEST.md"

        # Check if README.md is being created
        self.assertTrue(os.path.isfile(expected_file_path))

        # Check README.md content
        with open(expected_file_path, "r") as file:
            self.assertEqual(file.read().strip(), README_TEST)

        os.remove(os.getcwd() + "\README_TEST.md")

