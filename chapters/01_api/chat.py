"""调用 OpenAI Responses API 的最小示例。"""

import argparse
import os


def create_client():
    """延迟导入依赖，让 --help 在未安装 SDK 时仍可使用。"""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("缺少 OPENAI_API_KEY，请先设置环境变量。")
    return OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL"))


def main() -> None:
    parser = argparse.ArgumentParser(description="向 LLM 发送一条消息")
    parser.add_argument("prompt", help="用户问题")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    args = parser.parse_args()

    response = create_client().responses.create(
        model=args.model,
        instructions="你是一名严谨、简洁的中文助教。事实不确定时明确说明。",
        input=args.prompt,
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
