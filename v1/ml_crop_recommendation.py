import os
import json
import pickle
import numpy as np
from datetime import datetime
import requests
from geopy.geocoders import Nominatim
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CropPrediction:
    crop: str
    confidence_score: float
    confidence_percentage: float


@dataclass
class WeatherData:
    temperature: float
    humidity: float
    rainfall: float


class CropRecommendationModel:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.classes_ = None
    
    def load_model(self, model_path: str) -> bool:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as file:
                self.model = pickle.load(file)
            
            if not hasattr(self.model, 'predict') or not hasattr(self.model, 'predict_proba'):
                raise ValueError("Invalid model: missing predict or predict_proba methods")
            
            if hasattr(self.model, 'classes_'):
                self.classes_ = self.model.classes_
            else:
                raise ValueError("Model must have classes_ attribute")
            
            self.model_type = "random_forest"
            logger.info(f"Successfully loaded RandomForest model from {model_path}")
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, features: np.ndarray) -> List[str]:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self.model.predict_proba(features)


class LocationDataFetcher:
    def __init__(self, user_agent: str = "crop-recommendation-system"):
        self.geolocator = Nominatim(user_agent=user_agent, timeout=15)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
    
    def geocode_location(self, location_text: str) -> Optional[Tuple[float, float]]:
        try:
            location = self.geolocator.geocode(location_text, exactly_one=True)
            if location:
                return float(location.latitude), float(location.longitude)
        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
        return None
    
    def fetch_weather_data(self, lat: float, lon: float) -> Optional[WeatherData]:
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": ["temperature_2m", "relative_humidity_2m", "precipitation"],
                "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
                "forecast_days": 7,
                "timezone": "auto"
            }
            
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            current = data.get("current", {})
            daily = data.get("daily", {})
            
            temp_max = daily.get("temperature_2m_max", [])
            temp_min = daily.get("temperature_2m_min", [])
            precipitation = daily.get("precipitation_sum", [])
            
            if current.get("temperature_2m"):
                temperature = float(current["temperature_2m"])
            elif temp_max and temp_min:
                temperature = (sum(temp_max) + sum(temp_min)) / (2 * len(temp_max))
            else:
                raise ValueError("No temperature data available")
            
            if current.get("relative_humidity_2m"):
                humidity = float(current["relative_humidity_2m"])
            else:
                raise ValueError("No humidity data available")
            
            if precipitation:
                rainfall = sum(precipitation)
            elif current.get("precipitation"):
                rainfall = float(current["precipitation"]) * 30
            else:
                raise ValueError("No rainfall data available")
            
            return WeatherData(
                temperature=temperature,
                humidity=humidity,
                rainfall=rainfall
            )
            
        except Exception as e:
            logger.error(f"Weather data fetch failed: {e}")
            return None
    
    def fetch_soil_data(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        try:
            url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
            params = {
                "lat": lat,
                "lon": lon,
                "property": ["phh2o", "soc", "cec"],
                "depth": ["0-5cm"],
                "value": "mean"
            }
            
            response = self.session.get(url, params=params, timeout=25)
            response.raise_for_status()
            data = response.json()
            
            soil_properties = {}
            layers = data.get("properties", {}).get("layers", [])
            
            # Process each layer to extract and scale the mean values
            for layer in layers:
                prop_name = layer.get("name")
                depths = layer.get("depths", [])
                if not depths:
                    continue
                # SoilGrids stores values under depths[0]["values"]["mean"]
                mean_value = depths[0].get("values", {}).get("mean")
                if prop_name and mean_value is not None:
                    # Scale values that are multiplied by 10 in the API response
                    if prop_name in ("phh2o", "soc", "cec"):
                        soil_properties[prop_name] = mean_value / 10.0
                    else:
                        soil_properties[prop_name] = mean_value
            
            return soil_properties
            
        except Exception as e:
            logger.error(f"Soil data fetch failed: {e}")
            return None
    
    def estimate_npk_from_soil(self, soil_data: Dict[str, Any]) -> Tuple[float, float, float]:
        N, P, K = 60.0, 30.0, 35.0
        
        try:
            soc = soil_data.get("soc")
            if soc is not None:
                N = max(20.0, min(140.0, 30.0 + soc * 2.0))
            
            cec = soil_data.get("cec")
            if cec is not None:
                P = max(10.0, min(80.0, 15.0 + cec * 1.5))
                K = max(15.0, min(200.0, 20.0 + cec * 3.0))
                
        except Exception as e:
            logger.warning(f"NPK estimation error: {e}")
        
        return round(N, 1), round(P, 1), round(K, 1)


class CropRecommendationSystem:
    def __init__(self, model_path: str):
        self.model = CropRecommendationModel()
        self.data_fetcher = LocationDataFetcher()
        
        if not self.model.load_model(model_path):
            raise RuntimeError("Failed to load RandomForest model")
    
    def predict_from_parameters(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        temperature: float,
        humidity: float,
        ph: float,
        rainfall: float
    ) -> Dict[str, Any]:
        try:
            features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            top_5_indices = np.argsort(probabilities)[-5:][::-1]
            top_5_predictions = [
                CropPrediction(
                    crop=self.model.classes_[idx],
                    confidence_score=probabilities[idx],
                    confidence_percentage=round(probabilities[idx] * 100, 2)
                )
                for idx in top_5_indices
            ]
            
            return {
                "status": "success",
                "model_type": self.model.model_type,
                "recommended_crop": prediction,
                "top_5_recommendations": [
                    {
                        "crop": pred.crop,
                        "confidence_score": pred.confidence_score,
                        "confidence_percentage": pred.confidence_percentage
                    }
                    for pred in top_5_predictions
                ],
                "input_parameters": {
                    "nitrogen": nitrogen,
                    "phosphorus": phosphorus,
                    "potassium": potassium,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall
                },
                "metadata": {
                    "prediction_timestamp": datetime.now().isoformat(),
                    "total_classes": len(self.model.classes_)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "input_parameters": {
                    "nitrogen": nitrogen,
                    "phosphorus": phosphorus,
                    "potassium": potassium,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall
                }
            }
    
    def predict_from_location(
        self,
        location: str,
        parameter_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        parameter_overrides = parameter_overrides or {}
        
        try:
            coordinates = self.data_fetcher.geocode_location(location)
            if not coordinates:
                return {"status": "error", "error_message": "Failed to geocode location"}
            
            lat, lon = coordinates
            
            weather = self.data_fetcher.fetch_weather_data(lat, lon)
            if not weather:
                return {"status": "error", "error_message": "Failed to fetch weather data"}
            
            soil_data = self.data_fetcher.fetch_soil_data(lat, lon)
            if not soil_data:
                return {"status": "error", "error_message": "Failed to fetch soil data"}
            
            N, P, K = self.data_fetcher.estimate_npk_from_soil(soil_data)
            
            ph = soil_data.get("phh2o")
            if ph:
                ph = ph / 10.0
            else:
                return {"status": "error", "error_message": "Failed to get pH data"}
            
            final_params = {
                "nitrogen": parameter_overrides.get("nitrogen", N),
                "phosphorus": parameter_overrides.get("phosphorus", P),
                "potassium": parameter_overrides.get("potassium", K),
                "temperature": parameter_overrides.get("temperature", weather.temperature),
                "humidity": parameter_overrides.get("humidity", weather.humidity),
                "ph": parameter_overrides.get("ph", ph),
                "rainfall": parameter_overrides.get("rainfall", weather.rainfall)
            }
            
            result = self.predict_from_parameters(**final_params)
            
            if result["status"] == "success":
                result["location_info"] = {
                    "location": location,
                    "latitude": lat,
                    "longitude": lon
                }
                result["data_sources"] = {
                    "geocoding": "Nominatim",
                    "weather": "Open-Meteo API",
                    "soil": "ISRIC SoilGrids"
                }
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "location": location
            }


def main():
    print("Crop Recommendation System")
    print("=" * 50)
    
    model_path = "/Users/shubhamkumarmandal/Desktop/Farmer_assistant_bot/crop_recommendation_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        print("Please ensure the RandomForest.pkl file is in the current directory")
        return
    
    try:
        system = CropRecommendationSystem(model_path)
        
        print("\nTest 1: Direct Parameter Prediction")
        print("-" * 40)
        
        result1 = system.predict_from_parameters(
            nitrogen=90, phosphorus=42, potassium=43,
            temperature=20.9, humidity=82.0, ph=6.5, rainfall=203
        )
        print(json.dumps(result1, indent=2))
        
        print("\nTest 2: Location-based Prediction")
        print("-" * 40)
        
        result2 = system.predict_from_location(
            location="Kharagpur, West Bengal, India",
            parameter_overrides={"ph": 6.8}
        )
        print(json.dumps(result2, indent=2))
        
    except Exception as e:
        print(f"System initialization failed: {e}")


if __name__ == "__main__":
    main()