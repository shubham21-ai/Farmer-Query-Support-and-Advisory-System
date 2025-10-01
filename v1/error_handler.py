"""
Enhanced error handling and fallback mechanisms for the Farmer Assistant Bot
"""

import logging
import traceback
from typing import Dict, Any, Optional
from functools import wraps
import streamlit as st

logger = logging.getLogger(__name__)

class ErrorHandler:
    """Centralized error handling for the application"""
    
    @staticmethod
    def handle_api_error(error: Exception, service_name: str) -> Dict[str, Any]:
        """Handle API-related errors with graceful fallbacks"""
        error_msg = str(error)
        
        if "SSL" in error_msg or "certificate" in error_msg:
            return {
                "success": False,
                "error": f"{service_name} connection issue",
                "fallback_message": "Using offline data. Service will resume shortly.",
                "error_type": "ssl_error"
            }
        elif "timeout" in error_msg.lower():
            return {
                "success": False,
                "error": f"{service_name} timeout",
                "fallback_message": "Service is slow. Please try again in a moment.",
                "error_type": "timeout_error"
            }
        elif "api" in error_msg.lower() or "key" in error_msg.lower():
            return {
                "success": False,
                "error": f"{service_name} authentication issue",
                "fallback_message": "API service unavailable. Using alternative methods.",
                "error_type": "auth_error"
            }
        else:
            return {
                "success": False,
                "error": f"{service_name} error: {error_msg}",
                "fallback_message": "Using fallback system for this request.",
                "error_type": "general_error"
            }
    
    @staticmethod
    def handle_model_error(error: Exception, model_name: str) -> Dict[str, Any]:
        """Handle ML model-related errors"""
        error_msg = str(error)
        
        if "memory" in error_msg.lower() or "out of memory" in error_msg.lower():
            return {
                "success": False,
                "error": f"{model_name} memory issue",
                "fallback_message": "Using lightweight model. Results may be simplified.",
                "error_type": "memory_error"
            }
        elif "model" in error_msg.lower() and "not found" in error_msg.lower():
            return {
                "success": False,
                "error": f"{model_name} not available",
                "fallback_message": "Using alternative analysis method.",
                "error_type": "model_not_found"
            }
        else:
            return {
                "success": False,
                "error": f"{model_name} processing error",
                "fallback_message": "Using rule-based analysis instead.",
                "error_type": "model_error"
            }
    
    @staticmethod
    def display_error(error_info: Dict[str, Any]):
        """Display error in a user-friendly way"""
        error_type = error_info.get("error_type", "general_error")
        
        if error_type == "ssl_error":
            st.warning(f"🌐 {error_info['fallback_message']}")
        elif error_type == "timeout_error":
            st.warning(f"⏱️ {error_info['fallback_message']}")
        elif error_type == "auth_error":
            st.error(f"🔑 {error_info['fallback_message']}")
        elif error_type == "memory_error":
            st.warning(f"💾 {error_info['fallback_message']}")
        elif error_type == "model_not_found":
            st.warning(f"🤖 {error_info['fallback_message']}")
        else:
            st.error(f"⚠️ {error_info['fallback_message']}")
        
        # Log the actual error for debugging
        logger.error(f"Error: {error_info['error']}")

def safe_execute(func, fallback_result=None, service_name="Service"):
    """Safely execute a function with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {service_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            error_info = ErrorHandler.handle_api_error(e, service_name)
            ErrorHandler.display_error(error_info)
            
            return fallback_result or {
                "success": False,
                "error": str(e),
                "fallback_used": True
            }
    return wrapper

def with_fallback(fallback_func=None):
    """Decorator to add fallback functionality to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                
                if fallback_func:
                    try:
                        logger.info(f"Using fallback for {func.__name__}")
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {str(fallback_error)}")
                
                error_info = ErrorHandler.handle_api_error(e, func.__name__)
                ErrorHandler.display_error(error_info)
                
                return {
                    "success": False,
                    "error": str(e),
                    "fallback_attempted": fallback_func is not None
                }
        return wrapper
    return decorator

class FallbackData:
    """Provide fallback data when APIs are unavailable"""
    
    CROP_RECOMMENDATIONS = {
        "punjab": ["wheat", "rice", "cotton", "sugarcane"],
        "west bengal": ["rice", "jute", "potato", "mustard"],
        "maharashtra": ["cotton", "sugarcane", "rice", "wheat"],
        "uttar pradesh": ["wheat", "rice", "sugarcane", "potato"],
        "default": ["wheat", "rice", "maize", "cotton"]
    }
    
    WEATHER_DATA = {
        "temperature": 25.0,
        "humidity": 70.0,
        "rainfall": 50.0
    }
    
    MARKET_PRICES = {
        "wheat": {"min": 1800, "max": 2200, "modal": 2000},
        "rice": {"min": 2000, "max": 2800, "modal": 2400},
        "cotton": {"min": 5000, "max": 6500, "modal": 5800}
    }
    
    @staticmethod
    def get_crop_recommendations(location: str) -> list:
        """Get fallback crop recommendations"""
        location_lower = location.lower()
        for region, crops in FallbackData.CROP_RECOMMENDATIONS.items():
            if region in location_lower:
                return crops
        return FallbackData.CROP_RECOMMENDATIONS["default"]
    
    @staticmethod
    def get_weather_data(location: str) -> Dict[str, float]:
        """Get fallback weather data"""
        return FallbackData.WEATHER_DATA.copy()
    
    @staticmethod
    def get_market_prices(commodity: str) -> Dict[str, int]:
        """Get fallback market prices"""
        commodity_lower = commodity.lower()
        for crop, prices in FallbackData.MARKET_PRICES.items():
            if crop in commodity_lower:
                return prices
        return {"min": 1000, "max": 3000, "modal": 2000}

def create_progress_indicator(text="Processing..."):
    """Create a progress indicator for long-running operations"""
    return st.empty().info(f"⏳ {text}")

def update_progress(placeholder, text, progress=None):
    """Update progress indicator"""
    if progress is not None:
        placeholder.progress(progress, text=text)
    else:
        placeholder.info(f"⏳ {text}")

def clear_progress(placeholder):
    """Clear progress indicator"""
    placeholder.empty()
