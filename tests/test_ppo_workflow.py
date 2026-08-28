import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "chapters/05_alignment/ppo_workflow.py"
SPEC = importlib.util.spec_from_file_location("ppo_workflow", MODULE_PATH)
ppo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ppo)


class FakeBackend:
    def generate(self, prompts):
        return [f"answer:{prompt}" for prompt in prompts]

    def score(self, prompts, responses):
        return [1.0 for _ in responses]

    def ppo_step(self, prompts, responses, rewards):
        return {"mean_reward": sum(rewards) / len(rewards)}


class PPOWorkflowTests(unittest.TestCase):
    def test_batch_connects_all_three_stages(self):
        result = ppo.train_batch(FakeBackend(), ["a", "b"])
        self.assertEqual(result.rewards, [1.0, 1.0])
        self.assertEqual(result.metrics["mean_reward"], 1.0)

    def test_empty_batch_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能为空"):
            ppo.train_batch(FakeBackend(), [])


if __name__ == "__main__":
    unittest.main()
