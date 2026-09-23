import asyncio
import logging
from app.services.nvidia_provider import NvidiaAIProvider
from app.schemas.ai import AIRequest
from app.core.config import settings

logging.basicConfig(level=logging.INFO)

async def main():
    print("Initializing NVIDIA Provider...")
    provider = NvidiaAIProvider()
    print(f"Base URL: {provider.base_url}")
    print(f"Model ID Configuration: {provider.model}")
    print(f"API Key present: {bool(provider.api_key)}")
    
    request = AIRequest(
        system_prompt="", user_prompt="Return exactly the word OK.",
        temperature=0.6,
        max_tokens=16,
        model_id="openai/gpt-oss-20b"
    )
    
    print("\nSending direct request via NvidiaAIProvider...")
    try:
        response = await provider.generate(request)
        print(f"\nStatus: SUCCESS")
        print(f"Model returned: {response.model}")
        print(f"Content: {response.content}")
        print(f"Usage: {response.usage}")
    except Exception as e:
        print(f"\nStatus: FAILED")
        print(f"Exception: {e.__class__.__name__}")
        print(f"Details: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
