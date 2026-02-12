import argparse
import json
import subprocess
from functools import wraps
from typing import Callable
import os
import sys
from pathlib import Path

from openai import OpenAI

# Load .env file manually if it exists from root or app folder
for env_path in [".env", "app/.env"]:
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

API_KEY = os.getenv("KIMI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if API_KEY:
    API_KEY = API_KEY.strip()
    if API_KEY.startswith("Bearer "):
        API_KEY = API_KEY[7:]

# Set defaults based on the provider (NVIDIA, Moonshot, or OpenRouter)
if API_KEY and API_KEY.startswith("nvapi-"):
    BASE_URL = os.getenv("BASE_URL", default="https://integrate.api.nvidia.com/v1")
    MODEL = os.getenv("MODEL_NAME", default="moonshotai/kimi-k2.5")
elif os.getenv("KIMI_API_KEY"):
    BASE_URL = os.getenv("KIMI_BASE_URL", default="https://api.moonshot.ai/v1")
    MODEL = os.getenv("MODEL_NAME", default="kimi-k2.5")
else:
    BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
    MODEL = os.getenv("MODEL_NAME", default="anthropic/claude-haiku-4.5")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("API key not set. Please set KIMI_API_KEY or OPENROUTER_API_KEY.")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    while True:
        print(f"--- Calling AI ({MODEL})... waiting for response ---", file=sys.stderr)
        try:
            chat = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read and return the content of a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to read",
                                    }
                                },
                                "required": ["file_path"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "description": "Write content to a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to write to",
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The content to write to the file",
                                    },
                                },
                                "required": ["file_path", "content"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "Execute a shell command",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": "The command to execute",
                                    },
                                },
                                "required": ["command"],
                            },
                        },
                    },
                ],
                timeout=120,
            )
            print("--- Received Response ---", file=sys.stderr)
        except Exception as e:
            print(f"API Error: {e}", file=sys.stderr)
            break

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")
        if not chat.choices[0].message:
            raise RuntimeError("no message in response")
        message = chat.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
        )

        # You can use print statements as follows for debugging, they'll be visible when running tests.
        print("Logs from your program will appear here!", file=sys.stderr)

        if not message.tool_calls:
            if message.content:
                print(message.content)
            break
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"--- Executing Tool: {function_name} ({arguments.get('file_path') or arguments.get('command')}) ---", file=sys.stderr)
            if function_name == "Read":
                with open(arguments["file_path"], "r") as f:
                    content = f.read()
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": content,
                        }
                    )
            elif function_name == "Write":
                with open(arguments["file_path"], "w") as f:
                    f.write(arguments["content"])
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "File written successfully",
                        }
                    )
            elif function_name == "Bash":
                try:
                    result = subprocess.run(
                        arguments["command"],
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    output = result.stdout + result.stderr
                    if not output:
                        output = ""
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output,
                        }
                    )
                except Exception as e:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error executing command: {str(e)}",
                        }
                    )


if __name__ == "__main__":
    main()
