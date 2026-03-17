import os
from openai import OpenAI
from tools import search_tool
import re

SYSTEM_PROMPT = """
You are an intelligent reasoning agent. Solve problems using the following Thought -> Action -> Observation loop.


Thought: Think about what to do next. You should plan your steps carefully.
Action: Execute exactly one action. The ONLY action available to you is:
Search["your query here"]
Observation: The result of the search action.


Example:
User: What is the capital of France?
Thought: I need to find the capital of France.
Action: Search["Capital of France"]
Observation: The capital of France is Paris.
Thought: I have the answer.
The final answer is Paris.
"""



class ReActAgent:
    def __init__(self, system_prompt):
        self.system = system_prompt
        self.messages = [
            {"role": "system", "content": self.system}
        ]
        self.client = OpenAI()
    def construct_prompt(self, query):
        self.messages.append({"role": "user", "content": query})
    def execute(self):
        iteration = 0
        while iteration < 5:
            print(f"\n--- Iteration {iteration + 1} ---")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                stop=["Observation:"]
            )
            result = response.choices[0].message.content
            print(f"LLM輸出:\n{result}")
            self.messages.append({"role": "assistant", "content": result})
            
            if "Action:" not in result:
                print("== 找到最終答案，結束迴圈 ==")
                return result
            print("== 偵測到Action，準備執行工具 ==")
            match = re.search(r"Action:\s*Search\[['\"]?(.*?)['\"]?\]", result)
            if not match:
                print("== LLM格式錯誤或產生幻覺，強制回報錯誤讓他反思 ==")
                error_msg= "Observation: Invalid Action Format. You MUST use exactly Action: Search[\"query\"]."
                self.messages.append({"role": "user", "content": error_msg})
                iteration += 1
                continue
            query_str = match.group(1)
            print(f"準備搜尋關鍵字: {query_str}")
            search_result = search_tool(query_str)
            print(f"搜尋結果:\n{search_result}")
            observation_msg = f"Observation: {search_result}"
            self.messages.append({"role": "user", "content": observation_msg})
            iteration += 1
        return "達到最大迭代次數，結束執行"
