# 修复消息格式和安全问题
from openai import OpenAI
import os

class Deepseek:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key="输入API",
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.history = []

    def chat_stream(self, messages, temperature=1.2, max_tokens=1000):
        """修正消息格式处理"""
        if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
            raise ValueError("messages必须是由字典组成的列表")
        if not messages:  # 检查 messages 是否为空列表
            raise ValueError("messages不能为空列表")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        full_response = []
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
                full_response.append(content)

        # 将完整回复加入历史
        self.history.append({
            "role": "assistant",
            "content": "".join(full_response)
        })

import unittest
from unittest.mock import patch, MagicMock

class TestDeepseekChatStream(unittest.TestCase):
    def setUp(self):
        self.deepseek = Deepseek(model="deepseek-chat")
        self.mock_client = MagicMock()
        self.deepseek.client = self.mock_client

    def test_chat_stream_valid_input(self):
        # 模拟 OpenAI 的流式响应
        mock_response = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello, "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="world!"))]),
        ]
        self.mock_client.chat.completions.create.return_value = mock_response

        # 构造输入消息
        messages = [{"role": "user", "content": "Hello"}]

        # 调用 chat_stream 方法并收集输出
        response_chunks = list(self.deepseek.chat_stream(messages))

        # 验证输出
        self.assertEqual(response_chunks, ["Hello, ", "world!"])
        self.mock_client.chat.completions.create.assert_called_once_with(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )

    def test_chat_stream_invalid_input(self):
        # 测试无效输入（非列表或非字典消息）
        invalid_messages = "This is not a list of dictionaries"

        with self.assertRaises(ValueError):
            list(self.deepseek.chat_stream(invalid_messages))

    def test_chat_stream_empty_input(self):
        # 测试空输入
        empty_messages = []

        with self.assertRaises(ValueError):
            list(self.deepseek.chat_stream(empty_messages))

    def test_chat_stream_history_update(self):
        # 模拟 OpenAI 的流式响应
        mock_response = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello, "))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="world!"))]),
        ]
        self.mock_client.chat.completions.create.return_value = mock_response

        # 构造输入消息
        messages = [{"role": "user", "content": "Hello"}]

        # 调用 chat_stream 方法并收集输出
        list(self.deepseek.chat_stream(messages))

        # 验证历史记录是否正确更新
        self.assertEqual(len(self.deepseek.history), 1)
        self.assertEqual(self.deepseek.history[0]["role"], "assistant")
        self.assertEqual(self.deepseek.history[0]["content"], "Hello, world!")


if __name__ == "__main__":
    tst = TestDeepseekChatStream()
    tst.setUp()
    tst.test_chat_stream_empty_input()
    tst.test_chat_stream_valid_input()