import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "chapters/02_prompting/prompts.py"
SPEC = importlib.util.spec_from_file_location("prompts", MODULE_PATH)
prompts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prompts)


class PromptTests(unittest.TestCase):
    def test_every_pattern_contains_input(self):
        for pattern in prompts.PATTERNS:
            with self.subTest(pattern=pattern):
                self.assertIn("测试评论", prompts.build_prompt(pattern, "测试评论"))

    def test_few_shot_contains_examples(self):
        result = prompts.build_prompt("few_shot", "新评论")
        self.assertGreaterEqual(result.count("标签："), 4)

    def test_unknown_pattern_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "未知 pattern"):
            prompts.build_prompt("unknown", "text")


if __name__ == "__main__":
    unittest.main()
