#!/usr/bin/env python3
"""
API Key Setup Script for Farmer Query Support and Advisory System
This script helps you set up the required API keys for the application.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create or update the .env file with API key placeholders"""
    env_file = Path(".env")
    
    env_content = """# Required API Keys for Farmer Query Support and Advisory System
# Get your API keys from the following sources:

# 1. GOOGLE_API_KEY - Required for AI functionality
# Get from: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=your_gemini_api_key_here

# 2. TAVILY_API_KEY - Optional, for enhanced web search
# Get from: https://tavily.com/
TAVILY_API_KEY=your_tavily_api_key_here

# 3. GOOGLE_APPLICATION_CREDENTIALS - Optional, for Google Cloud services
# Path to your Google Cloud service account JSON file
GOOGLE_APPLICATION_CREDENTIALS=path_to_google_cloud_credentials.json
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created {env_file} with API key placeholders")
    return env_file

def check_api_keys():
    """Check if API keys are properly set"""
    print("\n🔍 Checking API Keys Status:")
    print("=" * 50)
    
    # Check GOOGLE_API_KEY
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and google_key != "your_gemini_api_key_here":
        print("✅ GOOGLE_API_KEY: Set")
    else:
        print("❌ GOOGLE_API_KEY: Not set or using placeholder")
    
    # Check TAVILY_API_KEY
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key and tavily_key != "your_tavily_api_key_here":
        print("✅ TAVILY_API_KEY: Set")
    else:
        print("⚠️  TAVILY_API_KEY: Not set (optional)")
    
    # Check GOOGLE_APPLICATION_CREDENTIALS
    gcp_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if gcp_creds and gcp_creds != "path_to_google_cloud_credentials.json":
        print("✅ GOOGLE_APPLICATION_CREDENTIALS: Set")
    else:
        print("⚠️  GOOGLE_APPLICATION_CREDENTIALS: Not set (optional)")

def print_setup_instructions():
    """Print detailed setup instructions"""
    print("\n📋 API Key Setup Instructions:")
    print("=" * 50)
    
    print("\n1. 🔑 GOOGLE_API_KEY (Required)")
    print("   - Go to: https://aistudio.google.com/app/apikey")
    print("   - Sign in with your Google account")
    print("   - Click 'Create API Key'")
    print("   - Copy the generated API key")
    print("   - Replace 'your_gemini_api_key_here' in .env file")
    
    print("\n2. 🌐 TAVILY_API_KEY (Optional - for web search)")
    print("   - Go to: https://tavily.com/")
    print("   - Sign up for a free account")
    print("   - Get your API key from the dashboard")
    print("   - Replace 'your_tavily_api_key_here' in .env file")
    
    print("\n3. ☁️  GOOGLE_APPLICATION_CREDENTIALS (Optional - for Google Cloud)")
    print("   - Go to: https://console.cloud.google.com/")
    print("   - Create a new project or select existing")
    print("   - Enable Speech-to-Text, Translation, and Text-to-Speech APIs")
    print("   - Create a service account and download JSON key file")
    print("   - Update the path in .env file")
    
    print("\n4. 🚀 After setting up keys:")
    print("   - Restart the Streamlit app")
    print("   - Run: streamlit run app.py")

def main():
    print("🌾 Farmer Query Support and Advisory System - API Setup")
    print("=" * 60)
    
    # Create .env file
    env_file = create_env_file()
    
    # Load environment variables
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    # Check current status
    check_api_keys()
    
    # Print instructions
    print_setup_instructions()
    
    print("\n✨ Setup complete! Follow the instructions above to add your API keys.")
    print("📁 Edit the .env file to add your actual API keys.")

if __name__ == "__main__":
    main()
