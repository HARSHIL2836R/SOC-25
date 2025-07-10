#!/usr/bin/env python3
"""
Interactive configuration helper for setting up API keys.
"""

import os
from pathlib import Path

def get_env_file_path():
    """Get the path to the .env file"""
    return Path(__file__).parent / ".env"

def read_env_file():
    """Read current .env file contents"""
    env_file = get_env_file_path()
    if env_file.exists():
        with open(env_file, 'r') as f:
            return f.read()
    return ""

def write_env_file(content):
    """Write content to .env file"""
    env_file = get_env_file_path()
    with open(env_file, 'w') as f:
        f.write(content)

def update_api_key(env_content, key_name, new_value):
    """Update or add an API key in the environment content"""
    lines = env_content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if line.startswith(f"{key_name}="):
            lines[i] = f"{key_name}={new_value}"
            updated = True
            break
    
    if not updated:
        lines.append(f"{key_name}={new_value}")
    
    return '\n'.join(lines)

def interactive_setup():
    """Interactive setup for API keys"""
    print("🔧 Research Agent API Configuration Helper")
    print("=" * 50)
    
    # Read current configuration
    env_content = read_env_file()
    
    print("\n📋 Current API Key Status:")
    print("-" * 30)
    
    # Check current keys
    current_groq = ""
    current_tavily = ""
    
    for line in env_content.split('\n'):
        if line.startswith("GROQ_API_KEY="):
            current_groq = line.split('=', 1)[1]
        elif line.startswith("TAVILY_API_KEY="):
            current_tavily = line.split('=', 1)[1]
    
    # Show current status
    if current_groq and current_groq != "your_groq_api_key_here":
        print("✅ GROQ_API_KEY: Configured")
    else:
        print("❌ GROQ_API_KEY: Not configured")
    
    if current_tavily and current_tavily != "your_tavily_api_key_here":
        print("✅ TAVILY_API_KEY: Configured")
    else:
        print("❌ TAVILY_API_KEY: Not configured")
    
    print("\n" + "=" * 50)
    
    # GROQ API Key setup
    print("\n🔑 GROQ API Key Setup (Required)")
    print("-" * 35)
    print("The GROQ API key is required for the LLM functionality.")
    print("Get your free key from: https://console.groq.com/keys")
    
    if current_groq and current_groq != "your_groq_api_key_here":
        update_groq = input(f"\nGROQ key is already configured. Update it? (y/N): ").lower()
        if update_groq == 'y':
            new_groq = input("Enter your new GROQ API key: ").strip()
            if new_groq:
                env_content = update_api_key(env_content, "GROQ_API_KEY", new_groq)
                print("✅ GROQ API key updated!")
    else:
        new_groq = input("\nEnter your GROQ API key: ").strip()
        if new_groq:
            env_content = update_api_key(env_content, "GROQ_API_KEY", new_groq)
            print("✅ GROQ API key configured!")
        else:
            print("⚠️  Skipping GROQ API key. The application may not work without it.")
    
    # Tavily API Key setup
    print("\n🌐 TAVILY API Key Setup (Optional)")
    print("-" * 37)
    print("The TAVILY API key enables internet search for finding similar papers.")
    print("Get your free key from: https://tavily.com/")
    print("(You can skip this if you only want to analyze local papers)")
    
    if current_tavily and current_tavily != "your_tavily_api_key_here":
        update_tavily = input(f"\nTAVILY key is already configured. Update it? (y/N): ").lower()
        if update_tavily == 'y':
            new_tavily = input("Enter your new TAVILY API key (or press Enter to skip): ").strip()
            if new_tavily:
                env_content = update_api_key(env_content, "TAVILY_API_KEY", new_tavily)
                print("✅ TAVILY API key updated!")
    else:
        new_tavily = input("\nEnter your TAVILY API key (or press Enter to skip): ").strip()
        if new_tavily:
            env_content = update_api_key(env_content, "TAVILY_API_KEY", new_tavily)
            print("✅ TAVILY API key configured!")
        else:
            print("⚠️  Skipping TAVILY API key. Internet search will be disabled.")
    
    # Save configuration
    try:
        write_env_file(env_content)
        print("\n💾 Configuration saved to .env file!")
        
        print("\n🚀 Setup Complete!")
        print("-" * 20)
        print("You can now run the application with:")
        print("   streamlit run app.py")
        print("\nOr test your configuration with:")
        print("   python test_setup.py")
        
    except Exception as e:
        print(f"\n❌ Error saving configuration: {e}")
        print("Please check file permissions and try again.")

def main():
    """Main function"""
    try:
        interactive_setup()
    except KeyboardInterrupt:
        print("\n\n👋 Configuration cancelled.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
