"""将 Hugging Face 模型仓库下载到确定的本地目录。"""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 Hugging Face 模型")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--revision", default="main", help="生产环境建议使用 commit hash")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.local_dir,
        token=os.getenv("HF_TOKEN") or None,
    )
    print(f"模型已下载到：{path}")


if __name__ == "__main__":
    main()
