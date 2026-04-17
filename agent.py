import asyncio
import os
import json
import re
from typing import Optional, List
from dotenv import load_dotenv
from browser_use import Agent, ChatOpenAI, BrowserSession as Browser, BrowserProfile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Load API keys
load_dotenv()

# 1. Patch ChatOpenAI to ensure compatibility with browser-use
class CompatibleChatOpenAI(ChatOpenAI):
    """
    Standard ChatOpenAI often lacks the 'provider' attribute that 
    browser-use Agent expects. This patch fixes it definitively.
    """
    @property
    def provider(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self.model

# Define Models
class ProductDetails(BaseModel):
    name: str
    price: str
    rating: float
    url: str

class QueryRefinement(BaseModel):
    is_vague: bool
    explanation: Optional[str] = None
    clarifying_question: Optional[str] = None
    refined_query: Optional[str] = None

# 2. Configure Global Browser instance with persistence
# user_data_dir ensures logins are saved. keep_alive=True keeps session open.
profile = BrowserProfile(
    headless=False, 
    keep_alive=True,
    user_data_dir=os.path.abspath(".browser_session")
)
browser = Browser(browser_profile=profile)

def parse_product_from_text(text: str) -> Optional[dict]:
    """Fallback parser to extract product details from Markdown/text."""
    try:
        data = {}
        # Try to find a JSON block in the text
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            raw_data = json.loads(json_match.group())
            # Normalize keys to lowercase for the rest of the script
            data = {k.lower(): v for k, v in raw_data.items()}
        else:
            # Manually extract using regex if it's Markdown-like
            name_match = re.search(r'(?i)\*\*?name\*\*?:\s*(.*)', text) or re.search(r'(?i)name:\s*(.*)', text)
            price_match = re.search(r'(?i)\*\*?price\*\*?:\s*([^ \n\r]*)', text) or re.search(r'(?i)price:\s*([^ \n\r]*)', text)
            rating_match = re.search(r'(?i)\*\*?rating\*\*?:\s*([\d\.\s/outf5]+)', text) or re.search(r'(?i)rating:\s*([\d\.\s/outf5]+)', text)
            url_match = re.search(r'\]\((https://www\.flipkart.com/[^\)]+)\)', text) or re.search(r'(https://www\.flipkart\.com/[^\s\n]+)', text)
            
            if name_match: data['name'] = name_match.group(1).strip()
            if price_match: data['price'] = price_match.group(1).strip()
            if rating_match: data['rating_str'] = rating_match.group(1).strip()
            if url_match: data['url'] = url_match.group(1).strip()

        # Clean up Rating (extract the first float/number found)
        rating_raw = data.get('rating') or data.get('rating_str')
        if rating_raw:
            num_match = re.search(r'([\d\.]+)', str(rating_raw))
            data['rating'] = float(num_match.group(1)) if num_match else 0.0
        
        if data.get('name') and data.get('price'):
            return data
    except Exception as e:
        print(f"⚠️  Parsing fallback failed: {e}")
    return None

async def refine_query(user_query: str) -> tuple[bool, str]:
    """
    Analyzes the user query for vagueness and asks for clarification if needed.
    Returns (should_proceed, message_or_refined_query).
    """
    llm = CompatibleChatOpenAI(model='gpt-4o')
    prompt = f"""
    Analyze the user's shopping query: "{user_query}"
    
    1. Is it too vague for a professional shopping search (e.g., "best phone", "good laptop")?
    2. Does it lack essential constraints like price range, brand preference, or specific use-case?
    
    If it's too vague, set is_vague=True and provide a polite clarifying_question that asks for 
    missing details (like budget, brand, or specific specs).
    
    If it's specific enough, set is_vague=False and provide a refined_query that encapsulates 
    the search intent clearly.
    """
    
    try:
        # 1. First, TRANSLATE the user query if it's describing a problem instead of a product
        translation_prompt = f"""
        The user said: "{user_query}"
        
        If this is a descriptive request (e.g. "something to hang clothes", "help me walk dirt free"),
        translate it into the specific product name they need (e.g. "Cloth Drying Stand", "Doormat").
        
        If it's already a product name (e.g. "iPhone 15", "Running Shoes"), return it as is.
        
        Return ONLY the product search term. No extra words.
        """
        res_trans = await llm.ainvoke(translation_prompt)
        translated_query = res_trans.content.strip().strip('"')
        
        # 2. Now check for specificity/budget using the TRANSLATED query
        vague_check_prompt = f"""
        Analyze this shopping query: '{translated_query}' (Original: '{user_query}')
        
        Is there a budget or price range mentioned? 
        - If NO, reply with: YES. Professional search requires a budget. Please ask: "Do you have a specific budget expectation or price range for {translated_query}?"
        - If YES, but the product itself is too broad (e.g. just 'shoes'), reply with: YES. And ask for brand/specs.
        - If YES and the product is specific, reply with: NO.
        
        Reply ONLY with YES/NO followed by the question if applicable.
        """
        res = await llm.ainvoke(vague_check_prompt)
        content = res.content.strip()
        
        if content.upper().startswith("YES"):
            # Extract just the question part
            msg = content[3:].strip().replace('Professional search requires a budget. Please ask: ', '')
            return False, msg
            
        # Return the TRANSLATED query for the actual search, not the user's raw description
        return True, translated_query
        
    except Exception as e:
        print(f"⚠️  Clarification check failed: {e}")
        return True, user_query

async def run_search_phase(user_query, exclude_items: List[str] = None):
    print(f"\n🕵️  Scout Agent: Searching for '{user_query}'...")
    if exclude_items:
        print(f"🚫 Excluding previously seen: {', '.join(exclude_items)}")
    
    llm = CompatibleChatOpenAI(model='gpt-4o')
    
    exclusion_instruction = ""
    if exclude_items:
        exclusion_instruction = f"\nNB: DO NOT suggest any of these products again: {', '.join(exclude_items)}. Look for different models or better specs."

    # ADVANCED INDUSTRY STANDARDS PROMPT
    search_task = f"""
1. Go to https://flipkart.com
2. Close any login popups.
3. Search for "{user_query}".
4. Find the "Best" product by applying these professional criteria:
   - RELIABILITY: Prioritize items with the 'Flipkart Assured' or 'Plus' badge.
   - RATING: Must be 4.0 stars or higher. Do not select items below 4.0 unless specifically asked for "cheap" options.
   - VALIDATION: Choose items with a significant number of reviews (ideally 500+). 
     Avoid 5-star items with only 1 or 2 reviews.
   - SPECS: For technical items (like laptops/phones), prioritize modern specs.
   - PRICE: Stay within the user's budget if mentioned.{exclusion_instruction}
5. IMPORTANT: Click on the chosen product to open its dedicated page.
6. Once on the product page, extract: Name, Price, Rating, and the CURRENT URL.
7. Return the final answer in the structured format.
"""

    agent = Agent(
        task=search_task,
        llm=llm,
        browser=browser,
        result_type=ProductDetails,
        use_vision=True,
        max_failures=5
    )

    try:
        history = await agent.run()
        
        # Method 1: Get structured output directly
        try:
            result = history.get_structured_output(ProductDetails)
            if result:
                # Basic cleanup: if URL is still a search URL, something went wrong
                if "flipkart.com/search" in result.url:
                     print("⚠️  Warning: Agent returned a search URL. Attempting to fix...")
                return result.model_dump()
        except Exception:
            pass
            
        # Method 2: Check final result string
        final_res = history.final_result()
        if final_res:
            parsed = parse_product_from_text(final_res)
            if parsed:
                return parsed
                    
        return None
    except Exception as e:
        print(f"❌ Search Error: {e}")
        return None

async def run_buy_phase(product_url):
    print(f"\n🛒 Buyer Agent: Adding to cart...")
    
    llm = CompatibleChatOpenAI(model='gpt-4o')
    
    buy_task = f"""
1. Navigate directly to {product_url}
2. Wait for the page to fully load, then scroll down slightly (0.5 pages) to bring the 'ADD TO CART' button into view.
3. Look for the primary action button that says 'ADD TO CART' or 'GO TO CART'. It may be YELLOW, ORANGE, WHITE, or any other color.
4. If you see variant options (like Size, Color, etc.) that are unselected, click on the first available option to enable the cart button.
5. Once the 'ADD TO CART' or 'GO TO CART' button is clearly visible and enabled, click it precisely.
6. Wait 2-3 seconds for the cart animation/confirmation.
7. Finally, navigate to https://www.flipkart.com/viewcart to show the cart contents.
8. Stop once you are on the cart page showing items.

CRITICAL: Do NOT click on "Specifications", "Description", or any other informational sections. Focus ONLY on the main 'ADD TO CART' or 'GO TO CART' button.
"""

    agent = Agent(
        task=buy_task,
        llm=llm,
        browser=browser,
        use_vision=True
    )

    try:
        await agent.run()
        print("\n✅ Successfully added to cart and navigated to your cart page!")
    except Exception as e:
        print(f"❌ Buy Error: {e}")

async def adjust_query_with_feedback(original_query: str, feedback: str) -> str:
    """Uses LLM to merge original query with user feedback for a better search."""
    llm = CompatibleChatOpenAI(model='gpt-4o')
    prompt = f"""
    The user originally searched for: "{original_query}"
    They saw a result but rejected it with this feedback: "{feedback}"
    
    Translate this into a single, optimized search query for Flipkart that addresses the feedback.
    Example:
    Original: "iPhone 15"
    Feedback: "too expensive, show me something around 60k"
    Result: "iPhone 15 under 60000"
    
    Return ONLY the new search string.
    """
    try:
        res = await llm.ainvoke(prompt)
        return res.content.strip().strip('"')
    except Exception:
        return f"{original_query} {feedback}"

async def main():
    print("🤖 browser-use Shopping Agent")
    print("-----------------------------")
    
    try:
        while True:
            user_query = input("\nWhat would you like to buy? (or 'exit'): ").strip()
            if user_query.lower() in ['exit', 'quit']:
                break
            
            if not user_query: continue

            suggested_names = [] # To prevent repeat suggestions

            # Outer search loop for retries
            while True:
                # --- Consultation Phase ---
                print(f"🧐 Analyzing your request...")
                should_proceed, response = await refine_query(user_query)
                
                if not should_proceed:
                    print(f"\n💡 {response}")
                    extra_info = input("Please provide more details (or press Enter to search anyway): ").strip()
                    if extra_info:
                        user_query = f"{user_query} {extra_info}"
                
                # --- Search Phase ---
                product = await run_search_phase(user_query, exclude_items=suggested_names)
                
                if not product or not product.get('name'):
                    print("❌ Failed to find a valid product. Please try a different query.")
                    break # Break inner loop to ask for a fresh query
                
                # Record this name so we don't suggest it again in a loop
                suggested_names.append(product.get('name'))
                    
                print("\n🔎 FOUND THE BEST MATCH:")
                print(f"   Name:   {product.get('name')}")
                print(f"   Price:  {product.get('price')}")
                print(f"   Rating: {product.get('rating')} ★")
                print(f"   URL:    {product.get('url')}")
                
                confirm = input("\nProceed to Add to Cart? (y/n, or give feedback to search again): ").lower().strip()
                if confirm == 'y' and product.get('url'):
                    await run_buy_phase(product.get('url'))
                    print("\nCheck the browser window for your cart!")
                    break # Task finished
                elif confirm in ['n', 'no', '']:
                    print("\nSearch reset.")
                    break # Start fresh
                else:
                    # Treat anything else as feedback (e.g., "too expensive", "different color")
                    print(f"🔄 Refining search based on feedback: '{confirm}'...")
                    user_query = await adjust_query_with_feedback(user_query, confirm)
                    # Continue inner loop to search again with refined query
    finally:
        # Close browser session when exiting the program
        await browser.stop()

if __name__ == "__main__":
    asyncio.run(main())
