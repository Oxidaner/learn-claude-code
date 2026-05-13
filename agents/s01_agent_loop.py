#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""

import os
import subprocess

try:
    import readline
    # 143 UTF-8 backspace fix for macOS libedit
    # 尝试导入 readline 模块（用于增强命令行输入体验）
    # 配置 UTF-8 字符正确处理（解决 macOS 兼容性问题）
    readline.parse_and_bind('set bind-tty-special-chars off')  #
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
    readline.parse_and_bind('set enable-meta-keybindings on')
except ImportError:
    # 如果导入失败（Windows 没有 readline），静默跳过
    pass


# Anthropic 官方 SDK，用于调用 Claude API
from anthropic import Anthropic
# 从 .env 文件加载环境变量
from dotenv import load_dotenv

# 从 .env 文件加载环境变量，override=True 表示覆盖已存在的环境变量。
load_dotenv(override=True)

# 如果设置了自定义 ANTHROPIC_BASE_URL（比如使用代理或本地服务）
# 则移除 ANTHROPIC_AUTH_TOKEN，避免认证冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 初始化 Anthropic 客户端
# 如果 ANTHROPIC_BASE_URL 为 None，使用官方 API
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 获取模型 ID：从环境变量读取要使用的模型名称。
MODEL = os.environ["MODEL_ID"]

# 系统提示词：
#    告诉模型它是一个编码 Agent
#    指明当前工作目录
#    要求使用 bash 解决任务，直接行动不要过多解释
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


# 工具定义:
#   定义了一个名为 bash 的工具
#   描述：运行 shell 命令
#   输入模式：需要一个 command 字符串参数
TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


def run_bash(command: str) -> str:
    # 定义危险命令列表
    # 包含 rm -rf /、sudo、shutdown、reboot、> /dev/ 等可能导致数据丢失或系统损坏的命令
    # 如果命令包含任何危险模式，直接拒绝执行
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 执行命令：
        # shell=True：通过 shell 执行
        # cwd=os.getcwd()：在当前工作目录执行
        # capture_output=True：捕获标准输出和错误
        # text=True：以文本形式返回
        # timeout=120：超时限制 120 秒
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)

        # 合并 stdout 和 stderr，移除首尾空格
        out = (r.stdout + r.stderr).strip()
        # 截取前 50000 字符，避免输出过长
        return out[:50000] if out else "(no output)"
    # 捕获超时异常，返回超时错误
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    # 捕获文件未找到或其他系统错误，返回错误信息
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})
        # If the model didn't call a tool, we're done
        if response.stop_reason != "tool_use":
            return
        # Execute each tool call, collect results
        results = []
        # 遍历响应内容中的所有块（block）
        for block in response.content:
            # 找到 tool_use 类型的块
            if block.type == "tool_use":
                # 打印命令（黄色显示）
                print(f"\033[33m$ {block.input['command']}\033[0m")
                # 执行 bash 命令
                output = run_bash(block.input["command"])
                # 打印输出前 200 字符
                print(output[:200])
                # 追加工具结果到结果列表
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": output})
        # 将所有工具结果作为用户消息添加到历史记录
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = [] # 存储对话历史
    while True: # 主循环，持续接收用户输入并处理
        try:
            # 显示青色提示符 s01 >>
            query = input("\033[36ms01 >> \033[0m")
        # 捕获 Ctrl+D (EOFError) 或 Ctrl+C (KeyboardInterrupt) 退出
        except (EOFError, KeyboardInterrupt):
            break
        # 退出条件：输入 q、exit 或空行时退出程序。
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 将用户输入添加到历史
        history.append({"role": "user", "content": query})
        # 调用 agent_loop 让模型处理
        agent_loop(history)
        # 获取模型回复
        response_content = history[-1]["content"]
        # 获取最后一条消息（模型的最终响应）
        # 如果是列表（包含多个块），遍历每个块，打印 text 属性，空行分隔
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(block.text)
        print()

        # 用户输入 → LLM 思考 → 调用工具 → 执行命令 → 返回结果 → LLM 继续思考 → ... → 最终回答
