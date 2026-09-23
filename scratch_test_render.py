import asyncio
import os
import httpx
import time
from jose import jwt

SUPABASE_JWT_SECRET = "YkW19Ok6w50DyGmSAojrLrm9kYu6op+edZBilXT9LXks2/eNvwNS9u5YC7YowMn9DG8zaUyindSH5KLSpyL3Kgw=="
RENDER_URL = "https://resume-app-backend-k2is.onrender.com"

async def main():
    print("1. Generating HS256 auth token locally...")
    payload = {
        "sub": "test-user-id",
        "role": "authenticated",
        "exp": int(time.time()) + 3600
    }
    token = jwt.encode(payload, SUPABASE_JWT_SECRET, algorithm="HS256")
    print(f"Token obtained: {token[:10]}...")
    
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    async with httpx.AsyncClient(timeout=200.0) as client:
        print("\n=== POST /ai/test-provider ===")
        r1 = await client.post(f"{RENDER_URL}/ai/test-provider", headers=headers, json={})
        print(f"Status: {r1.status_code}")
        print(f"Body: {r1.text[:200]}")
        
        print("\n=== POST /ai/chat ===")
        r2 = await client.post(f"{RENDER_URL}/ai/chat", headers=headers, json={
            "user_id": "test", 
            "messages": [{"role": "user", "content": "Return the word OK."}]
        })
        print(f"Status: {r2.status_code}")
        print(f"Body: {r2.text[:200]}")
        
        print("\n=== POST /ai/resume/polish ===")
        r3 = await client.post(f"{RENDER_URL}/ai/resume/polish", headers=headers, json={
            "original_content": "Did some things.", 
            "section_type": "experience", 
            "instruction": "Make it sound professional", 
            "target_role": "Software Engineer"
        })
        print(f"Status: {r3.status_code}")
        print(f"Body: {r3.text[:200]}")
        
        print("\n=== POST /ats/parse ===")
        r4 = await client.post(f"{RENDER_URL}/ats/parse", headers=headers, json={
            "text": "Software Engineer with 5 years of Python experience."
        })
        print(f"Status: {r4.status_code}")
        print(f"Body: {r4.text[:200]}")
        
        print("\n=== POST /ats/analyze ===")
        r5 = await client.post(f"{RENDER_URL}/ats/analyze", headers=headers, json={
            "resume_text": "Software Engineer with 5 years of Python experience. Built REST APIs.",
            "job_description": "Looking for a Python backend engineer with REST API experience.",
            "job_title": "Backend Engineer",
            "industry": "Tech"
        })
        print(f"Status: {r5.status_code}")
        if r5.status_code == 200:
            data = r5.json()
            print(f"Score: {data.get('overall_score')}")
            print(f"Model used: {data.get('model')}")
        else:
            print(f"Body: {r5.text[:200]}")

if __name__ == "__main__":
    asyncio.run(main())
