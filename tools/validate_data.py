"""在启动昂贵训练前，快速检查课程 JSONL 数据的结构与内容。"""

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


class DataValidationError(ValueError):
    """包含文件行号、便于定位的数据错误。"""


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"{path}:{line_number}: JSON 格式错误：{exc.msg}") from exc
            if not isinstance(row, dict):
                raise DataValidationError(f"{path}:{line_number}: 每行必须是 JSON object")
            yield line_number, row


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_sft(row: dict[str, Any]) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise DataValidationError("messages 必须是至少包含两条消息的数组")
    allowed_roles = {"system", "user", "assistant"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DataValidationError(f"messages[{index}] 必须是 object")
        if message.get("role") not in allowed_roles:
            raise DataValidationError(f"messages[{index}].role 不合法")
        if not _non_empty_string(message.get("content")):
            raise DataValidationError(f"messages[{index}].content 不能为空")
    if not any(message["role"] == "assistant" for message in messages):
        raise DataValidationError("messages 必须包含 assistant 回答")


def validate_preference(row: dict[str, Any]) -> None:
    for field in ("prompt", "chosen", "rejected"):
        if not _non_empty_string(row.get(field)):
            raise DataValidationError(f"{field} 必须是非空字符串")
    if row["chosen"].strip() == row["rejected"].strip():
        raise DataValidationError("chosen 和 rejected 不能相同")


def validate_file(path: Path, data_type: str) -> int:
    validator = validate_sft if data_type == "sft" else validate_preference
    count = 0
    for line_number, row in read_jsonl(path):
        try:
            validator(row)
        except DataValidationError as exc:
            raise DataValidationError(f"{path}:{line_number}: {exc}") from exc
        count += 1
    if count == 0:
        raise DataValidationError(f"{path}: 文件中没有数据")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="校验 SFT 或偏好数据 JSONL")
    parser.add_argument("data_type", choices=("sft", "preference"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        count = validate_file(args.path, args.data_type)
    except (DataValidationError, OSError) as exc:
        parser.exit(1, f"校验失败：{exc}\n")
    print(f"校验通过：{args.path}，共 {count} 条数据")


if __name__ == "__main__":
    main()
