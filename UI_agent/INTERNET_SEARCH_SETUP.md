# Research Paper Analysis Agent - Internet Search Setup Guide

## 🚀 Quick Start

The internet search functionality requires a free Tavily API key. Follow these steps to enable it:

### Step 1: Get API Keys

#### Required: GROQ API Key
1. Visit https://console.groq.com/keys
2. Sign up for a free account
3. Create a new API key
4. Copy the key

#### Optional: Tavily API Key (for Internet Search)
1. Visit https://tavily.com/
2. Sign up for a free account
3. Get your API key from the dashboard
4. Copy the key

### Step 2: Configure Your Environment

#### Option A: Automatic Configuration (Recommended)
Run the configuration helper:
```bash
python setup_config.py
```

#### Option B: Manual Configuration
1. Copy the example environment file:
   ```bash
   copy .env.example .env
   ```

2. Edit the `.env` file and replace the placeholder values:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   TAVILY_API_KEY=your_actual_tavily_api_key_here
   ```

### Step 3: Test Your Setup
```bash
python test_setup.py
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

## 🔍 Internet Search Features

Once configured with a Tavily API key, you can:

- **Find Similar Papers**: Ask "Find papers similar to this one"
- **Recent Research**: Ask "What are recent developments in this field?"
- **Comparative Analysis**: Ask "Show me comparative studies on this topic"
- **Survey Papers**: Ask "Find survey papers related to this research"

## 🎬 Step-by-Step Demo: Getting Tavily API Key

### 1. Visit Tavily Website
Go to https://tavily.com/ and click "Get Started" or "Sign Up"

### 2. Create Account
- Enter your email address
- Create a password
- Verify your email if required

### 3. Access Dashboard
- Log in to your account
- Navigate to the API section or dashboard
- Look for "API Keys" or "Credentials"

### 4. Generate API Key
- Click "Create New API Key" or similar button
- Copy the generated key (it looks like: `tvly-xxxxxxxxxxxxxxxxxxxx`)

### 5. Add to Configuration
- Open your `.env` file in the UI_agent folder
- Replace `your_tavily_api_key_here` with your actual key:
  ```
  TAVILY_API_KEY=tvly-your-actual-key-here
  ```

### 6. Restart Application
- Stop the Streamlit app (Ctrl+C)
- Run `streamlit run app.py` again
- Check the sidebar for "✅ Internet search ready"

## 🎯 What Internet Search Enables

With Tavily API configured, you can ask powerful research questions:

### Find Similar Papers
- **Input**: "Find papers similar to this research on transformers"
- **Result**: Recent papers from arXiv, Google Scholar, and academic databases

### Discover Recent Work
- **Input**: "What are the latest developments in this field?"
- **Result**: Current research trends and recent publications

### Comparative Analysis
- **Input**: "Compare this approach with recent methods"
- **Result**: Side-by-side analysis with related work

### Survey and Reviews
- **Input**: "Find survey papers on this topic"
- **Result**: Comprehensive review papers and state-of-the-art summaries

## 🛠️ Troubleshooting

### Internet Search Not Working?

1. **Check API Key**: Ensure `TAVILY_API_KEY` is set in your `.env` file
2. **Verify Connection**: Run `python test_setup.py` to test your configuration
3. **Check Spelling**: Make sure the key in `.env` is exactly `TAVILY_API_KEY`
4. **Restart Application**: After updating `.env`, restart the Streamlit app

### Common Issues

#### "Tavily package not installed"
```bash
pip install tavily-python
```

#### "API key not configured"
- Check that your `.env` file exists in the same directory as `app.py`
- Verify the API key is not still the placeholder value
- Make sure there are no extra spaces around the key

#### "API connection failed"
- Verify your API key is correct
- Check your internet connection
- Ensure your Tavily account is active

### Status Indicators

The app shows your internet search status:

- ✅ **Internet search ready**: Everything configured correctly
- 🔑 **API key not configured**: Need to add Tavily API key
- ❌ **Connection failed**: Check API key and internet connection

## 🎯 Usage Examples

### Questions That Trigger Internet Search:
- "Find similar papers to this research"
- "What are recent works in this area?"
- "Show me related studies"
- "Find comparative analysis papers"
- "Search for survey papers on this topic"

### Local-Only Questions:
- "What is the main contribution of this paper?"
- "Explain the methodology used"
- "What are the results and conclusions?"
- "Summarize the related work section"

## 📁 File Structure

```
UI_agent/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration settings
├── research_chain.py     # LangGraph research agent
├── setup_config.py       # Interactive configuration helper
├── test_setup.py         # Setup testing script
├── .env                  # Your API keys (create this)
├── .env.example          # Example environment file
└── requirements.txt      # Python dependencies
```

## 🆘 Getting Help

If you're still having issues:

1. Run the test script: `python test_setup.py`
2. Check the configuration: `python setup_config.py`
3. Look at the status display in the app sidebar
4. Verify all files exist and API keys are correct

## 🔒 Security Notes

- Never commit your `.env` file to version control
- Keep your API keys secure and private
- The `.env` file should contain real keys, not placeholder values
- Both GROQ and Tavily offer free tiers with generous limits
