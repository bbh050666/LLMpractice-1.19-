"""常见 Prompt 模式：构造逻辑与 API 调用相互独立，便于测试。"""

import argparse
import os

PATTERNS = ("zero_shot", "few_shot", "role_constraints", "structured_reasoning")


def build_prompt(pattern: str, text: str) -> str:
    if pattern == "zero_shot":
        return f"判断评论情感，只输出 positive、neutral 或 negative。\n评论：{text}\n标签："
    if pattern == "few_shot":
        return (
            "判断评论情感，只输出标签。\n"
            "评论：物流很快，包装也很好。\n标签：positive\n"
            "评论：一般般，没有惊喜。\n标签：neutral\n"
            "评论：第二天就坏了。\n标签：negative\n"
            f"评论：{text}\n标签："
        )
    if pattern == "role_constraints":
        return (
            "角色：你是电商质检员。\n任务：判断评论情感。\n"
            "约束：只能输出 positive、neutral、negative 之一；不要解释。\n"
            f"评论：{text}\n输出："
        )
    if pattern == "structured_reasoning":
        return (
            "分析下面的需求。先列出最多 3 条可验证的关键依据，再给出明确结论；"
            "不要虚构信息。\n"
            f"需求：{text}\n输出格式：\n依据：\n- ...\n结论：..."
        )
    raise ValueError(f"未知 pattern：{pattern}；可选值：{', '.join(PATTERNS)}")


def run(prompt: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("使用 --run 时必须设置 OPENAI_API_KEY。")
    response = OpenAI(base_url=os.getenv("OPENAI_BASE_URL")).responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), input=prompt
    )
    return response.output_text


def main() -> None:
    parser = argparse.ArgumentParser(description="比较 Prompt 构造模式")
    parser.add_argument("--pattern", choices=PATTERNS, default="zero_shot")
    parser.add_argument("--text", default="外观不错，但续航和宣传的不一样。")
    parser.add_argument("--run", action="store_true", help="实际调用 API；默认只打印 Prompt")
    args = parser.parse_args()
    prompt = build_prompt(args.pattern, args.text)
    print(run(prompt) if args.run else prompt)


if __name__ == "__main__":
    main()
