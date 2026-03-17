import os
from dotenv import load_dotenv
from agent import ReActAgent, SYSTEM_PROMPT

load_dotenv()

if __name__ == "__main__":
    agent = ReActAgent(SYSTEM_PROMPT)
    user_question = "Who is the CEO of the startup 'Morphic' AI search?"
    agent.construct_prompt(user_question)
    print("目前Agent的記憶歷史:")
    for msg in agent.messages:
        print(f"[{msg['role']}]: {msg['content'][:100]}...")

agent.execute()