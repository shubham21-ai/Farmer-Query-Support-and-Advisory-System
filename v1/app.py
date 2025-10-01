import os
import io
import sys
import json
import tempfile
import logging
from typing import Optional, Dict, Any
import streamlit as st
from contextlib import redirect_stdout

# Import all our agents and components
from router_agent import RouterAgent, AgentType
from synthesizer_agent import SynthesizerAgent
from google_apis import GoogleAPIs, WHISPER_AVAILABLE, GTTS_AVAILABLE, ALTERNATIVE_SPEECH_AVAILABLE

# Import specialized agents
from disease_diagnosis import CropDiseaseAgent
from schemes import AgriSchemesAgent
from crop_recommendation_agent import CropRecommenderAgent
from price_detection_agent import MarketPriceDetectionAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FarmerAssistantPipeline:
    """Main pipeline that orchestrates all agents and APIs"""
    
    def __init__(self):
        try:
            # Initialize all components
            self.router = RouterAgent()
            self.synthesizer = SynthesizerAgent()
            self.google_apis = GoogleAPIs()
            
            # Initialize specialized agents
            self.disease_agent = CropDiseaseAgent()
            self.schemes_agent = AgriSchemesAgent()
            self.crop_agent = CropRecommenderAgent()
            self.price_agent = MarketPriceDetectionAgent()
            
            logger.info("Farmer Assistant Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def process_query(self, query: str, image_path: Optional[str] = None, 
                     source_language: str = "hi-IN", target_language: str = "hi-IN") -> Dict[str, Any]:
        """
        Process a farmer query through the complete pipeline
        
        Args:
            query: Farmer's query (in English or original language)
            image_path: Optional image file path for disease diagnosis
            source_language: Source language code for TTS
            target_language: Target language code for response
            
        Returns:
            Complete processing results
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Step 1: Route the query
            routing_result = self.router.route_query(query, has_image=image_path is not None)
            logger.info(f"Routed to: {routing_result.agent_type.value}")
            
            # Step 2: Execute appropriate agent(s)
            agent_results = {}
            
            if routing_result.agent_type == AgentType.DISEASE_DIAGNOSIS:
                result = self.disease_agent.analyze_disease(query, image_path)
                agent_results["disease_diagnosis"] = result.to_dict() if hasattr(result, 'to_dict') else result
                
            elif routing_result.agent_type == AgentType.CROP_RECOMMENDATION:
                result = self.crop_agent.respond(query)
                agent_results["crop_recommendation"] = result.to_dict() if hasattr(result, 'to_dict') else result
                
            elif routing_result.agent_type == AgentType.SCHEMES:
                result = self.schemes_agent.search_and_extract_schemes(query)
                agent_results["schemes"] = result.to_dict() if hasattr(result, 'to_dict') else result
                
            elif routing_result.agent_type == AgentType.PRICE_DETECTION:
                result = self.price_agent.get_market_analysis(query)
                agent_results["price_detection"] = {"market_analysis": result}
                
            elif routing_result.agent_type == AgentType.MULTIPLE:
                # Execute multiple agents based on extracted parameters
                params = routing_result.extracted_parameters
                
                # Check what agents to run based on keywords
                if any(keyword in query.lower() for keyword in ['crop', 'grow', 'plant', 'cultivate']):
                    result = self.crop_agent.respond(query)
                    agent_results["crop_recommendation"] = result.to_dict() if hasattr(result, 'to_dict') else result
                
                if any(keyword in query.lower() for keyword in ['disease', 'pest', 'sick', 'treatment']):
                    result = self.disease_agent.analyze_disease(query, image_path)
                    agent_results["disease_diagnosis"] = result.to_dict() if hasattr(result, 'to_dict') else result
                
                if any(keyword in query.lower() for keyword in ['scheme', 'subsidy', 'government', 'benefit']):
                    result = self.schemes_agent.search_and_extract_schemes(query)
                    agent_results["schemes"] = result.to_dict() if hasattr(result, 'to_dict') else result
                
                if any(keyword in query.lower() for keyword in ['price', 'rate', 'market', 'mandi']):
                    result = self.price_agent.get_market_analysis(query)
                    agent_results["price_detection"] = {"market_analysis": result}
            
            # Step 3: Synthesize results
            synthesized_response = self.synthesizer.synthesize_results(
                original_query=query,
                agent_results=agent_results
            )
            
            # Step 4: Translate response to target language
            translated_response = synthesized_response
            if target_language and not target_language.startswith("en"):
                logger.info(f"Translating response to {target_language}")
                try:
                    # Translate summary
                    summary_translation = self.google_apis.translate_text(
                        text=synthesized_response.summary,
                        target_language=target_language,
                        source_language="en"
                    )
                    
                    # Translate detailed analysis
                    analysis_translation = self.google_apis.translate_text(
                        text=synthesized_response.detailed_analysis,
                        target_language=target_language,
                        source_language="en"
                    )
                    
                    # Translate recommendations
                    translated_recommendations = []
                    for rec in synthesized_response.actionable_recommendations:
                        rec_translation = self.google_apis.translate_text(
                            text=rec,
                            target_language=target_language,
                            source_language="en"
                        )
                        if rec_translation["success"]:
                            translated_recommendations.append(rec_translation["translated_text"])
                        else:
                            translated_recommendations.append(rec)
                    
                    # Create translated response object
                    if summary_translation["success"]:
                        translated_response = type(synthesized_response)(
                            summary=summary_translation["translated_text"],
                            detailed_analysis=analysis_translation["translated_text"] if analysis_translation["success"] else synthesized_response.detailed_analysis,
                            actionable_recommendations=translated_recommendations,
                            confidence_score=synthesized_response.confidence_score,
                            sources=synthesized_response.sources,
                            timestamp=synthesized_response.timestamp,
                            agent_results=synthesized_response.agent_results
                        )
                        logger.info(f"Translation successful to {target_language}")
                    
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    logger.warning("Using English response")
            
            # Step 5: Convert response to speech (optional)
            audio_result = None
            if target_language and not target_language.startswith("en"):
                audio_result = self.google_apis.process_response_to_speech(
                    text=translated_response.summary,
                    target_language=target_language
                )
            
            return {
                "success": True,
                "routing_result": routing_result.to_dict(),
                "agent_results": agent_results,
                "synthesized_response": translated_response.to_dict(),
                "original_response": synthesized_response.to_dict() if translated_response != synthesized_response else None,
                "audio_response": audio_result,
                "processing_metadata": {
                    "agents_used": list(agent_results.keys()),
                    "routing_confidence": routing_result.confidence,
                    "synthesis_confidence": synthesized_response.confidence_score,
                    "target_language": target_language,
                    "translation_applied": translated_response != synthesized_response
                }
            }
            
        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "routing_result": None,
                "agent_results": {},
                "synthesized_response": None,
                "audio_response": None
            }
    
    def process_voice_query(self, audio_file_path: str, image_path: Optional[str] = None,
                           source_language: str = "hi-IN", target_language: str = "hi-IN") -> Dict[str, Any]:
        """
        Process voice query through complete pipeline
        
        Args:
            audio_file_path: Path to audio file
            image_path: Optional image file path
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Complete processing results including voice processing
        """
        try:
            logger.info(f"Processing voice query from: {audio_file_path}")
            
            # Step 1: Speech to Text and Translation
            voice_result = self.google_apis.process_voice_query(audio_file_path, source_language)
            
            if not voice_result["success"]:
                return {
                    "success": False,
                    "error": f"Voice processing failed: {voice_result.get('error', 'Unknown error')}",
                    "voice_result": voice_result
                }
            
            # Step 2: Process the translated query
            translated_query = voice_result["translated_text"]
            processing_result = self.process_query(
                query=translated_query,
                image_path=image_path,
                source_language=source_language,
                target_language=target_language
            )
            
            # Add voice processing results
            processing_result["voice_result"] = voice_result
            processing_result["original_query"] = voice_result["original_text"]
            processing_result["translated_query"] = translated_query
            
            return processing_result
            
        except Exception as e:
            logger.error(f"Voice query processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "voice_result": None,
                "processing_result": None
            }

class CaptureStdout:
    """Helper class to capture stdout for logging"""
    def __init__(self, container):
        self.container = container
        self.placeholder = container.empty()
        self.output = []
    
    def write(self, text):
        if text.strip():
            self.output.append(text)
            try:
                self.placeholder.code(''.join(self.output), language="text")
            except Exception:
                try:
                    self.placeholder = self.container.empty()
                    self.placeholder.code(''.join(self.output), language="text")
                except:
                    pass
    
    def flush(self):
        try:
            if self.output:
                self.placeholder.code(''.join(self.output), language="text")
        except:
            pass

def main():
    """Main Streamlit application with professional UI"""
    st.set_page_config(
        page_title="KisanSaathi - Smart Farming Assistant", 
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom styles
    from styles import load_css, create_header, create_feature_card, create_info_box, create_status_indicator, create_card
    load_css()
    
    # Create beautiful header
    create_header()
    
    # Initialize pipeline with enhanced caching
    @st.cache_resource(show_spinner=False)
    def initialize_pipeline():
        """Initialize the pipeline with caching for better performance"""
        try:
            with st.spinner("🚀 Initializing AI systems..."):
                return FarmerAssistantPipeline()
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            logger.error(f"Pipeline initialization failed: {str(e)}")
            return None
    
    # Cache frequently used data
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_language_options():
        """Get language options with caching"""
        return {
            "hi-IN": "🇮🇳 Hindi",
            "en-US": "🇺🇸 English", 
            "bn-IN": "🇧🇩 Bengali",
            "ta-IN": "🇮🇳 Tamil",
            "te-IN": "🇮🇳 Telugu",
            "mr-IN": "🇮🇳 Marathi",
            "gu-IN": "🇮🇳 Gujarati",
            "kn-IN": "🇮🇳 Kannada",
            "ml-IN": "🇮🇳 Malayalam",
            "pa-IN": "🇮🇳 Punjabi"
        }
    
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_crop_examples():
        """Get example queries with caching"""
        return [
            "What crops should I grow in Punjab during monsoon?",
            "My tomato plants have yellow spots, what's wrong?",
            "Current wheat prices in Maharashtra",
            "Government schemes for organic farming",
            "Weather forecast for crop planning in Delhi"
        ]
    
    pipeline = initialize_pipeline()
    
    if pipeline is None:
        st.error("❌ Pipeline initialization failed. Please check your API keys and dependencies.")
        st.stop()
    
    # Professional Sidebar
    with st.sidebar:
        st.markdown("### 🌱 AI Specialists")
        
        # Feature info
        st.info("🌾 **Crop Health** - Disease diagnosis & treatment")
        st.info("🌡️ **Weather Advisory** - Real-time weather insights")
        st.info("📊 **Market Prices** - Live commodity prices")
        st.info("🏛️ **Gov Schemes** - Agricultural subsidies")
        
        st.markdown("---")
        
        # Language settings with enhanced UX
        st.markdown("### 🌐 Language Settings")
        language_options = get_language_options()
        language_keys = list(language_options.keys())
        language_labels = list(language_options.values())
        
        source_language = st.selectbox(
            "Input Language",
            language_keys,
            format_func=lambda x: language_options[x],
            index=1,  # Default to English
            help="Choose your preferred input language"
        )
        
        target_language = st.selectbox(
            "Response Language", 
            language_keys,
            format_func=lambda x: language_options[x],
            index=1,  # Default to English  
            help="Choose your preferred response language"
        )
        
        st.markdown("---")
        
        # Input method selection
        st.markdown("### 📝 Input Method")
        input_type = st.radio(
            "How would you like to ask your question?",
            ["💬 Text Query", "🎤 Voice Query"],
            help="Choose your preferred input method"
        )
    
    # Welcome section for new users
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True
    
    if st.session_state.show_welcome:
        st.info("""
        🎯 **How can I help you today?**
        
        I'm your intelligent farming companion, ready to assist with:
        
        - 🌾 **Crop Recommendations** - Best crops for your location and season
        - 🔍 **Disease Diagnosis** - Identify plant diseases from images
        - 📋 **Government Schemes** - Agricultural subsidies and programs
        - 💰 **Market Prices** - Real-time commodity price information
        
        Simply ask your question below in text or voice!
        """)
        
        if st.button("Got it! Let's start 🚀"):
            st.session_state.show_welcome = False
            st.rerun()
    
    # Main content area
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        st.subheader("💬 Ask Your Question")
        
        # Text input with example suggestions
        if input_type == "💬 Text Query":
            query = st.text_area(
                "What would you like to know?",
                placeholder="Type your agricultural question here...",
                height=120,
                help="Ask anything about crops, diseases, weather, prices, or government schemes"
            )
            
            # Quick example buttons
            st.markdown("**💡 Try these examples:**")
            examples = get_crop_examples()
            cols = st.columns(len(examples))
            
            for i, example in enumerate(examples):
                with cols[i]:
                    if st.button(f"📝 {example[:25]}...", key=f"example_{i}", help=example):
                        st.session_state.query_input = example
                        st.rerun()
            
            # Use session state for query if set by example button
            if hasattr(st.session_state, 'query_input'):
                query = st.session_state.query_input
                del st.session_state.query_input
            
            audio_file = None
            
        else:  # Voice Query
            st.info("🎤 **Voice Input** - Upload an audio file with your agricultural question in any supported language.")
            
            audio_file = st.file_uploader(
                "Choose your audio file:",
                type=["wav", "mp3", "m4a", "flac"],
                help="Supported formats: WAV, MP3, M4A, FLAC"
            )
            
            query = ""
            if audio_file is not None:
                # Save uploaded audio to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(audio_file.read())
                    audio_file_path = tmp_file.name
                
                # Show audio player
                st.audio(audio_file.read(), format=f"audio/{audio_file.name.split('.')[-1]}")
                
                # Process voice query
                if st.button("🎤 Process Voice Query", type="primary"):
                    with st.spinner("Processing voice query..."):
                        result = pipeline.process_voice_query(
                            audio_file_path=audio_file_path,
                            source_language=source_language,
                            target_language=target_language
                        )
                        
                        if result["success"]:
                            st.success("✅ Voice query processed successfully!")
                            
                            # Display original and translated text
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"**Original ({source_language}):** {result.get('original_query', 'N/A')}")
                            with col2:
                                st.info(f"**Translated (English):** {result.get('translated_query', 'N/A')}")
                            
                            # Process the results
                            display_results(result)
                            
                            # Play audio response if available
                            if result.get("audio_response") and result["audio_response"]["success"]:
                                audio_file_path = result["audio_response"]["audio_file"]
                                if os.path.exists(audio_file_path):
                                    st.audio(audio_file_path, format="audio/wav")
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Image upload
        st.info("📷 **Image Analysis (Optional)** - Upload a clear image of your crop or plant for AI-powered disease diagnosis and analysis.")
        
        image_file = st.file_uploader(
            "Choose an image file:",
            type=["png", "jpg", "jpeg"],
            help="Supported formats: PNG, JPG, JPEG (max 10MB)"
        )
        
        image_path = None
        if image_file is not None:
            try:
                img_bytes = image_file.read()
                st.image(img_bytes, caption="📸 Uploaded Image for Analysis", use_container_width=True)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(img_bytes)
                    image_path = tmp_file.name
                
                st.success("✅ Image uploaded successfully! It will be analyzed along with your query.")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                image_path = None
        
        # Process button for text queries with improved UX
        if input_type == "💬 Text Query":
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                process_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
            
            if process_button:
                if query.strip():
                    # Create progress indicators
                    progress_container = st.container()
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("🧠 Analyzing your query...")
                            progress_bar.progress(20)
                            
                            status_text.text("🤖 Routing to appropriate AI specialist...")
                            progress_bar.progress(40)
                            
                            status_text.text("📊 Processing with AI models...")
                            progress_bar.progress(60)
                            
                            result = pipeline.process_query(
                                query=query,
                                image_path=image_path,
                                source_language=source_language,
                                target_language=target_language
                            )
                            
                            status_text.text("✨ Finalizing response...")
                            progress_bar.progress(80)
                            
                            if result["success"]:
                                progress_bar.progress(100)
                                status_text.text("✅ Complete!")
                                
                                # Hide progress after a moment
                                import time
                                time.sleep(1)
                                progress_container.empty()
                                
                                st.success("✅ Query processed successfully!")
                                display_results(result)
                                
                                # Play audio response if available
                                if result.get("audio_response") and result["audio_response"]["success"]:
                                    audio_file_path = result["audio_response"]["audio_file"]
                                    if os.path.exists(audio_file_path):
                                        st.audio(audio_file_path, format="audio/wav")
                            else:
                                progress_container.empty()
                                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            progress_container.empty()
                            st.error(f"❌ An unexpected error occurred: {str(e)}")
                            logger.error(f"Query processing error: {str(e)}")
                            
                else:
                    st.warning("⚠️ Please enter your agricultural question above")
    
    with col_right:
        # System Status
        st.subheader("📊 System Status")
        st.success("🟢 Online")
        st.success("🤖 AI Ready")
        st.info(f"**Language:** {language_options[source_language]} → {language_options[target_language]}")
        st.info(f"**Input:** {input_type}")
        
        st.markdown("---")
        
        # Available Services
        st.subheader("🤖 AI Services")
        st.markdown("""
        - 🧠 **Router Agent** - Intelligent query routing
        - 🌾 **Crop Advisor** - ML-based recommendations
        - 🔍 **Disease Expert** - Image-based diagnosis
        - 🏛️ **Scheme Finder** - Government programs
        - 💰 **Price Tracker** - Market analysis
        - 🔄 **Smart Synthesizer** - Result integration
        """)

def display_results(result: Dict[str, Any]):
    """Display processing results in a user-friendly format"""
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 Analysis", "🎯 Recommendations", "📊 Details"])
    
    synthesized = result.get("synthesized_response", {})
    routing = result.get("routing_result", {})
    agent_results = result.get("agent_results", {})
    
    with tab1:
        st.subheader("📋 Response Summary")
        
        if synthesized:
            st.write(f"**Summary:** {synthesized.get('summary', 'No summary available')}")
            st.write(f"**Confidence:** {synthesized.get('confidence_score', 0):.2f}")
            
            if synthesized.get('key_insights'):
                st.write("**Key Insights:**")
                for insight in synthesized['key_insights']:
                    st.write(f"• {insight}")
        
        if routing:
            st.write(f"**Routed to:** {routing.get('agent_type', 'Unknown')} agent")
            st.write(f"**Routing Confidence:** {routing.get('confidence', 0):.2f}")
    
    with tab2:
        st.subheader("🔍 Detailed Analysis")
        
        if synthesized and synthesized.get('detailed_analysis'):
            st.write(synthesized['detailed_analysis'])
        else:
            st.info("No detailed analysis available")
    
    with tab3:
        st.subheader("🎯 Actionable Recommendations")
        
        if synthesized and synthesized.get('actionable_recommendations'):
            for i, recommendation in enumerate(synthesized['actionable_recommendations'], 1):
                st.write(f"{i}. {recommendation}")
        else:
            st.info("No specific recommendations available")
        
        if synthesized and synthesized.get('next_steps'):
            st.write("**Next Steps:**")
            for step in synthesized['next_steps']:
                st.write(f"• {step}")
        
        if synthesized and synthesized.get('risk_factors'):
            st.write("**Risk Factors:**")
            for risk in synthesized['risk_factors']:
                st.write(f"⚠️ {risk}")
    
    with tab4:
        st.subheader("📊 Technical Details")
        
        # Show routing information
        if routing:
            st.write("**Routing Information:**")
            st.json(routing)
        
        # Show agent results
        if agent_results:
            st.write("**Agent Results:**")
            for agent_name, agent_result in agent_results.items():
                st.write(f"**{agent_name.replace('_', ' ').title()}:**")
                st.json(agent_result)
        
        # Show processing metadata
        metadata = result.get("processing_metadata", {})
        if metadata:
            st.write("**Processing Metadata:**")
            st.json(metadata)

if __name__ == "__main__":
    main()
