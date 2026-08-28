import importlib.util
import pathlib
import tempfile
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "tools/validate_data.py"
SPEC = importlib.util.spec_from_file_location("validate_data", MODULE_PATH)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(validator)


class DataValidationTests(unittest.TestCase):
    def test_course_demo_files_are_valid(self):
        root = pathlib.Path(__file__).parents[1]
        self.assertEqual(validator.validate_file(root / "data/sft_demo.jsonl", "sft"), 3)
        self.assertEqual(
            validator.validate_file(root / "data/preferences_demo.jsonl", "preference"), 3
        )

    def test_error_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "bad.jsonl"
            path.write_text('{"messages": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(validator.DataValidationError, r"bad\.jsonl:1"):
                validator.validate_file(path, "sft")

    def test_identical_preferences_are_rejected(self):
        with self.assertRaisesRegex(validator.DataValidationError, "不能相同"):
            validator.validate_preference({"prompt": "p", "chosen": "x", "rejected": "x"})


if __name__ == "__main__":
    unittest.main()
