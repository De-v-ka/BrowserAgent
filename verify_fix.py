import asyncio
from agent import get_llm
from browser_use import Agent, BrowserSession as Browser, BrowserProfile
from langchain_openai import ChatOpenAI
import os

async def test_init():
    print("Testing LLM initialization...")
    llm = get_llm()
    print(f"LLM type: {type(llm)}")
    
    # Simulate what browser-use does
    def dummy_ainvoke(): pass
    setattr(llm, 'ainvoke', dummy_ainvoke)
    print("Successfully set 'ainvoke' on LLM instance.")
    
    # Try creating an Agent instance
    print("Creating Agent instance...")
    profile = BrowserProfile(headless=True)
    browser = Browser(browser_profile=profile)
    agent = Agent(
        task="Test task",
        llm=llm,
        browser=browser
    )
    print("Agent instance created successfully.")
    await browser.stop()

if __name__ == "__main__":
    asyncio.run(test_init())
