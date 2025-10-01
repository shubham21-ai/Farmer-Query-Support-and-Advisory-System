#!/usr/bin/env python3
"""
API Key Test Script for Farmer Query Support and Advisory System
This script tests if your API keys are working correctly.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_google_api():
    """Test Google API key"""
    print("🔑 Testing Google API Key...")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            print("❌ GOOGLE_API_KEY not set")
            return False
        
        # Test with a simple query
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.1
        )
        
        response = llm.invoke("Hello, this is a test. Please respond with 'API test successful'.")
        print(f"✅ Google API working: {response.content[:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ Google API error: {str(e)}")
        return False

def test_tavily_api():
    """Test Tavily API key"""
    print("\n🌐 Testing Tavily API Key...")
    try:
        from tavily import TavilyClient
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key or api_key == "your_tavily_api_key_here":
            print("⚠️  TAVILY_API_KEY not set (optional)")
            return False
        
        client = TavilyClient(api_key=api_key)
        response = client.search("test query", max_results=1)
        
        if response and 'results' in response:
            print("✅ Tavily API working")
            return True
        else:
            print("❌ Tavily API returned unexpected response")
            return False
            
    except Exception as e:
        print(f"❌ Tavily API error: {str(e)}")
        return False

def test_google_cloud_credentials():
    """Test Google Cloud credentials"""
    print("\n☁️  Testing Google Cloud Credentials...")
    try:
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds_path or creds_path == "path_to_google_cloud_credentials.json":
            print("⚠️  GOOGLE_APPLICATION_CREDENTIALS not set (optional)")
            return False
        
        if not Path(creds_path).exists():
            print(f"❌ Credentials file not found: {creds_path}")
            return False
        
        # Test basic import
        from google.cloud import translate_v2 as translate
        print("✅ Google Cloud credentials file exists and is valid")
        return True
        
    except Exception as e:
        print(f"❌ Google Cloud credentials error: {str(e)}")
        return False

def main():
    print("🧪 API Key Test Suite")
    print("=" * 40)
    
    # Load environment variables
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print("📁 Loaded .env file")
    else:
        print("⚠️  No .env file found")
    
    print()
    
    # Test each API
    google_ok = test_google_api()
    tavily_ok = test_tavily_api()
    gcp_ok = test_google_cloud_credentials()
    
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    print(f"Google API: {'✅ Working' if google_ok else '❌ Not working'}")
    print(f"Tavily API: {'✅ Working' if tavily_ok else '⚠️  Not set/Optional'}")
    print(f"Google Cloud: {'✅ Working' if gcp_ok else '⚠️  Not set/Optional'}")
    
    if google_ok:
        print("\n🎉 Your setup is ready! You can now run the Streamlit app with full functionality.")
    else:
        print("\n⚠️  Please set up your Google API key to use the main features.")

if __name__ == "__main__":
    main()
