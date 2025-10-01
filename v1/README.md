# 🌾 Farmer Assistant Bot
![Uploading Screenshot 2025-09-28 at 3.41.12 PM.png…]()

A comprehensive agricultural intelligence system that provides farmers with crop recommendations, disease diagnosis, government scheme information, and market price analysis through an intelligent conversational interface with voice support.

## 🎯 Features

### Core Capabilities
- **🤖 Intelligent Routing**: Automatically routes farmer queries to appropriate specialized agents
- **🌱 Crop Recommendations**: ML-based crop suggestions using weather, soil, and market data
- **🔍 Disease Diagnosis**: Image-based plant disease identification and treatment recommendations
- **📋 Government Schemes**: Information about agricultural subsidies, loans, and government programs
- **💰 Price Detection**: Real-time market price analysis and comparison
- **🎤 Voice Support**: Speech-to-text and text-to-speech in multiple Indian languages
- **🔄 Smart Synthesis**: Combines results from multiple agents into coherent responses

### Language Support
- **Input Languages**: Hindi, English, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi
- **Output Languages**: Same as input languages with automatic translation

## 🏗️ System Architecture

```
Farmer Query (Voice/Text) → Google Speech-to-Text → Translation → Router Agent
                                                                    ↓
Specialized Agents ← Router Decision ← Query Analysis
     ↓                    ↓                    ↓                    ↓
Crop Recommendation   Disease Diagnosis   Government Schemes   Price Detection
                                                                    ↓
                    Synthesizer Agent ← All Agent Results
                                 ↓
                    Final Response → Translation → Text-to-Speech → Farmer
```

## 🤖 Agent Components

### 1. Router Agent (`router_agent.py`)
- **Purpose**: Intelligently routes farmer queries to appropriate specialized agents
- **Features**:
  - Natural language understanding
  - Multi-agent routing for complex queries
  - Parameter extraction from queries
  - Confidence scoring for routing decisions

### 2. Crop Recommendation Agent (`crop_recommendation_agent.py`)
- **Purpose**: Provides ML-based crop recommendations
- **Features**:
  - Uses Random Forest model trained on agricultural data
  - Considers soil parameters (N, P, K, pH)
  - Weather data integration
  - Location-based recommendations
  - Market profitability analysis

### 3. Disease Diagnosis Agent (`disease_diagnosis.py`)
- **Purpose**: Identifies plant diseases from images and text descriptions
- **Features**:
  - Vision Transformer (ViT) model for image classification
  - Fallback analysis for non-image queries
  - Treatment and prevention recommendations
  - Market search for treatment options

### 4. Government Schemes Agent (`schemes.py`)
- **Purpose**: Provides information about agricultural government schemes
- **Features**:
  - Real-time search for relevant schemes
  - Structured information extraction
  - Eligibility criteria and application process
  - Contact information and official sources

### 5. Price Detection Agent (`price_detection_agent.py`)
- **Purpose**: Analyzes market prices and trading opportunities
- **Features**:
  - Integration with Indian government market APIs
  - Price comparison across different mandis
  - Market trend analysis
  - Arbitrage opportunity identification

### 6. Synthesizer Agent (`synthesizer_agent.py`)
- **Purpose**: Combines outputs from multiple agents into coherent responses
- **Features**:
  - Intelligent information integration
  - Conflict resolution between different sources
  - Farmer-friendly output formatting
  - Actionable recommendation prioritization

### 7. Google APIs Integration (`google_apis.py`)
- **Purpose**: Handles speech processing and translation
- **Features**:
  - Google Cloud Speech-to-Text
  - Google Cloud Translation
  - Google Cloud Text-to-Speech
  - Fallback to alternative speech APIs
  - Multi-language support

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Google Cloud Platform account (for Google APIs)
- Tavily API key (for web search)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Farmer_assistant_bot
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables
Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional (for enhanced functionality)
TAVILY_API_KEY=your_tavily_api_key_here
GOOGLE_APPLICATION_CREDENTIALS=path_to_google_cloud_credentials.json
```

### Step 4: Google Cloud Setup (Optional)
For enhanced speech and translation features:

1. Create a Google Cloud Project
2. Enable the following APIs:
   - Cloud Speech-to-Text API
   - Cloud Translation API
   - Cloud Text-to-Speech API
3. Create service account credentials
4. Download the JSON key file
5. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable

## 🎮 Usage

### Running the Application
```bash
streamlit run app.py
```

### Using the Web Interface

1. **Choose Input Method**:
   - Text Query: Type your agricultural question
   - Voice Query: Upload an audio file with your question

2. **Select Languages**:
   - Source Language: Language of your input
   - Response Language: Language for the output

3. **Upload Image (Optional)**:
   - Upload crop/disease images for analysis

4. **Process Query**:
   - Click "Process Query" or "Process Voice Query"
   - View results in organized tabs

### Example Queries

**Crop Recommendations**:
- "What crops should I grow in Punjab this season?"
- "I have sandy loam soil with pH 7.2, what should I plant?"

**Disease Diagnosis**:
- "My tomato plants have yellow spots on leaves"
- Upload image of diseased plant

**Government Schemes**:
- "What government schemes are available for organic farming?"
- "How can I get crop insurance?"

**Market Prices**:
- "What are the current wheat prices in Haryana?"
- "Compare rice prices between different mandis"

## 🧪 Testing

Run the integration tests to verify all components:

```bash
python test_integration.py
```

This will test:
- Module imports
- Router agent functionality
- Google APIs integration
- Specialized agents
- Synthesizer agent
- Complete pipeline integration

## 📊 API Reference

### Router Agent
```python
from router_agent import RouterAgent

router = RouterAgent()
result = router.route_query("What crops should I grow?")
print(result.agent_type.value)  # crop_recommendation
```

### Synthesizer Agent
```python
from synthesizer_agent import SynthesizerAgent

synthesizer = SynthesizerAgent()
result = synthesizer.synthesize_results(query, agent_results)
print(result.summary)
```

### Google APIs
```python
from google_apis import GoogleAPIs

google_apis = GoogleAPIs()
# Speech to text
result = google_apis.speech_to_text("audio.wav", "hi-IN")
# Translation
result = google_apis.translate_text("Hello", "hi", "en")
# Text to speech
result = google_apis.text_to_speech("Hello", language_code="hi-IN")
```

## 🔧 Configuration

### Model Configuration
- **Router Agent**: Uses Gemini 2.0 Flash for routing decisions
- **Synthesizer Agent**: Uses Gemini 2.0 Flash for response synthesis
- **Crop Recommendation**: Uses trained Random Forest model
- **Disease Diagnosis**: Uses Vision Transformer model

### Language Configuration
Supported language codes:
- `hi-IN`: Hindi
- `en-US`: English
- `bn-IN`: Bengali
- `ta-IN`: Tamil
- `te-IN`: Telugu
- `mr-IN`: Marathi
- `gu-IN`: Gujarati
- `kn-IN`: Kannada
- `ml-IN`: Malayalam
- `pa-IN`: Punjabi

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure `GOOGLE_API_KEY` is set correctly
   - Check API key permissions

2. **Import Errors**:
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+)

3. **Google Cloud APIs Not Working**:
   - Verify Google Cloud credentials
   - Check API quotas and billing

4. **Speech Recognition Issues**:
   - Ensure audio file format is supported (WAV, MP3, M4A, FLAC)
   - Check microphone permissions

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

### Model Optimization
- Use GPU acceleration for Vision Transformer model
- Implement model caching for faster responses
- Use batch processing for multiple queries

### API Optimization
- Implement request caching
- Use connection pooling for external APIs
- Set appropriate timeouts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Cloud Platform for speech and translation APIs
- Hugging Face for pre-trained models
- LangChain for agent framework
- Streamlit for web interface
- Indian government APIs for agricultural data

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the test results for component status

---

**Built with ❤️ for Indian Farmers**
