# 🌾 KisanSaathi - Smart Agricultural Assistant

**Your Intelligent Agricultural Companion Powered by AI**

An advanced AI-powered agricultural advisory system that helps farmers with crop recommendations, disease diagnosis, weather information, market prices, and government schemes - all in multiple Indian languages.

## 🌟 Features

### 🤖 **AI-Powered Services**
- **Crop Recommendations** - ML-based suggestions using weather, soil, and market data
- **Disease Diagnosis** - Image-based plant disease identification using Vision Transformer
- **Weather Information** - Real-time weather data and forecasts
- **Market Prices** - Live commodity prices from Indian mandis
- **Government Schemes** - Agricultural subsidies and program information

### 🌐 **Multi-Language Support**
Supports 10 Indian languages:
- 🇮🇳 Hindi (hi-IN)
- 🇺🇸 English (en-US)
- 🇮🇳 Malayalam (ml-IN)
- 🇮🇳 Bengali (bn-IN)
- 🇮🇳 Tamil (ta-IN)
- 🇮🇳 Telugu (te-IN)
- 🇮🇳 Marathi (mr-IN)
- 🇮🇳 Gujarati (gu-IN)
- 🇮🇳 Kannada (kn-IN)
- 🇮🇳 Punjabi (pa-IN)

### 🎤 **Voice Features**
- **Speech-to-Text** - Voice input using Whisper
- **Text-to-Speech** - Audio output using gTTS
- **Multi-language voice support**

## 🏗️ Architecture

```
User Query (Text/Voice/Image)
           ↓
    Router Agent (Gemini)
           ↓
┌─────────┴─────────┐
│  Specialized Agents │
├────────────────────┤
│ • Crop Advisor     │
│ • Disease Expert   │
│ • Scheme Finder    │
│ • Price Tracker    │
└──────────┬─────────┘
           ↓
   Synthesizer Agent
           ↓
    Translation (Gemini)
           ↓
  Text-to-Speech (gTTS)
           ↓
    Response to User
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Google API Key (Gemini)
- Tavily API Key (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/Kisan_Saathi_Web.git
cd Kisan_Saathi_Web
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Get API keys from:
- Google Gemini: https://aistudio.google.com/app/apikey
- Tavily: https://tavily.com/

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the app**
Open your browser and go to: http://localhost:8501

## 📊 Tech Stack

- **Framework**: Streamlit
- **AI/LLM**: Google Gemini 2.0 Flash
- **ML Models**: Random Forest, Vision Transformer (ViT)
- **Speech**: Whisper (STT), gTTS (TTS)
- **Search**: Tavily API
- **Agent Framework**: LangChain

## 🎯 Usage Examples

### Text Query
```
Question: "What crops should I grow in Punjab during monsoon?"
Response: AI-powered recommendations based on location, weather, and soil
```

### Image Analysis
```
Upload: Plant disease image
Question: "What is this disease?"
Response: Disease identification, symptoms, treatment, and prevention
```

### Voice Input
```
Upload: Audio file in Hindi/Malayalam/other languages
Response: Transcribed, processed, and answered in preferred language
```

## 📁 Project Structure

```
Kisan_Saathi_Web/
├── app.py                          # Main Streamlit application
├── router_agent.py                 # Query routing logic
├── synthesizer_agent.py            # Response synthesis
├── google_apis.py                  # Translation & voice APIs
├── crop_recommendation_agent.py    # Crop advisor
├── disease_diagnosis.py            # Disease expert
├── schemes.py                      # Government schemes
├── price_detection_agent.py        # Price tracker
├── crop_recommendation_tool.py     # ML crop tools
├── get_price_tool.py              # Price fetching tools
├── styles.py                       # UI styling
├── error_handler.py               # Error handling utilities
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### Language Settings
- Configure input and response languages in the sidebar
- Translation powered by Google Gemini
- Supports all major Indian languages

### Model Configuration
- Router: Gemini 2.0 Flash
- Synthesizer: Gemini 2.0 Flash  
- Disease Detection: Vision Transformer
- Crop Recommendation: Random Forest

## 🌟 Key Features

### ✅ Intelligent Routing
Automatically routes queries to appropriate specialists based on content

### ✅ Multi-Agent System
- Router Agent - Query analysis and routing
- Crop Recommender - ML-based crop suggestions
- Disease Diagnoser - Image-based disease identification
- Scheme Finder - Government program information
- Price Detector - Market price analysis
- Synthesizer - Combines all agent outputs

### ✅ Professional UI
- Clean, modern interface
- Green agricultural theme
- Responsive design
- Progress indicators
- Error handling

## 🐛 Troubleshooting

### API Key Issues
- Ensure `GOOGLE_API_KEY` is valid and not expired
- Free tier: 50 requests/minute limit
- Get new key from: https://aistudio.google.com/app/apikey

### Translation Slow
- Check API quota limits
- Consider upgrading to paid tier for higher limits

### SSL Certificate Errors
- Already handled with fallback mechanisms
- App continues working with cached location data

## 📈 Performance

- Caching enabled for pipeline initialization
- Session state management for better UX
- Progress indicators for long operations
- Optimized model loading

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational and agricultural development purposes.

## 🙏 Acknowledgments

- Google Gemini for AI capabilities
- Hugging Face for ML models
- LangChain for agent framework
- Streamlit for web interface
- Indian Government APIs for agricultural data

## 📞 Support

For issues or questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review terminal logs for detailed error messages

---

**Built with ❤️ for Indian Farmers** 🌾

**KisanSaathi** - Empowering Agriculture with AI