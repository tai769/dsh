"""第 06 章：供提示词组装 Agent 使用的安全四则运算工具。"""

from __future__ import annotations

from math import isfinite
from typing import Any

from client import Tool


def _evaluate(source: str) -> float:
    """使用递归下降解析器计算算术表达式。"""
    tokens = _tokenize(source)
    position = 0

    def peek() -> str | None:
        return tokens[position] if position < len(tokens) else None

    def take() -> str:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("表达式意外结束")
        token = tokens[position]
        position += 1
        return token

    def parse_expression() -> float:
        value = parse_term()
        while peek() in ("+", "-"):
            operator = take()
            right = parse_term()
            value = value + right if operator == "+" else value - right
        return value

    def parse_term() -> float:
        value = parse_factor()
        while peek() in ("*", "/"):
            operator = take()
            right = parse_factor()
            if operator == "*":
                value *= right
            else:
                if right == 0:
                    raise ValueError("除数为零")
                value /= right
        return value

    def parse_factor() -> float:
        token = take()
        if token == "(":
            value = parse_expression()
            if take() != ")":
                raise ValueError("缺少右括号")
            return value
        if token == "-":
            return -parse_factor()
        value = float(token)
        if not isfinite(value):
            raise ValueError("数字超出有限范围")
        return value

    result = parse_expression()
    if position != len(tokens):
        raise ValueError(f"表达式包含未解析内容: {tokens[position]!r}")
    if not isfinite(result):
        raise ValueError("计算结果超出有限范围")
    return result


def _tokenize(source: str) -> list[str]:
    """将表达式拆成数字、运算符和括号。"""
    tokens: list[str] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
        elif char in "+-*/()":
            tokens.append(char)
            index += 1
        elif char.isdigit() or char == ".":
            number = ""
            while index < len(source) and (
                source[index].isdigit() or source[index] == "."
            ):
                number += source[index]
                index += 1
            tokens.append(number)
        else:
            raise ValueError(f"非法字符: {char!r}")
    return tokens


def _run_calculator(args: dict[str, Any]) -> str:
    expression = args.get("expression")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("参数 expression 必须是非空字符串")
    return str(_evaluate(expression))


calculator = Tool(
    name="calculator",
    description="计算一个四则运算表达式，支持 + - * / 与括号，例如 '1+2*3'",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "要计算的数学表达式，例如 '1+2*(3-1)'",
            }
        },
        "required": ["expression"],
    },
    execute=_run_calculator,
)
