# Copilot Instructions for Farmer Assistant Bot

## Project Overview
This repository is a multi-agent system for agricultural support, providing crop disease diagnosis, crop recommendations, and government scheme information. The main user interface is built with Streamlit (`app.py`). Agents use LLMs (Google Gemini via LangChain), image classification (ViT), and web search (Tavily).

## Key Components
- `app.py`: Streamlit UI orchestrating agent calls and user interactions.
- `diagnosis.py`: Crop disease diagnosis agent. Uses ViT image classification and Gemini LLM. Requires `GOOGLE_API_KEY` and optionally `TAVILY_API_KEY` in `.env`.
- `schemes.py`: Agent for searching and extracting government agricultural schemes using Tavily and Gemini.
- `crop_recommendation_agent.py`: (Stub) Intended for crop recommendation logic.
- `ml_crop_recommendation.py`, `train.py`: ML model training and prediction for crop recommendation (Random Forest, scikit-learn).
- `crop-recommendation-rf/src/`: Contains scripts for ML model training (`train_model.py`), prediction (`predict.py`), and data utilities (`utils/data_loader.py`).

## Developer Workflows
- **Run UI:**
  ```bash
  streamlit run app.py
  ```
- **Train ML Model:**
  ```bash
  python train.py
  # Or use scripts in crop-recommendation-rf/src/
  ```
- **Test Disease Agent:**
  ```bash
  python diagnosis.py
  ```
- **Test Schemes Agent:**
  ```bash
  python schemes.py
  ```

## Environment Setup
- Install Python dependencies:
  ```bash
  pip install -r requirements.txt
  # For ML scripts: pip install pandas scikit-learn matplotlib seaborn
  ```
- Set API keys in `.env`:
  - `GOOGLE_API_KEY` (required for LLM features)
  - `TAVILY_API_KEY` (optional, for market/scheme search)

## Patterns & Conventions
- Agents use Pydantic models for structured output.
- LLM prompts enforce strict JSON output for downstream parsing.
- Image classification uses HuggingFace ViT models (see `diagnosis.py`).
- ML model is saved as `crop_recommendation_model.pkl` and loaded for predictions.
- All agent classes provide a `run()` or `analyze_*()` method for main logic.
- Fallbacks are implemented for missing API keys or model errors.

## Integration Points
- LLM: Google Gemini via LangChain (`langchain-google-genai`)
- Search: Tavily API (`langchain_community.tools.tavily_search`)
- ML: scikit-learn, transformers, torch
- UI: Streamlit

## Example: Adding a New Agent
- Create a new file (e.g., `weather_advice.py`) with a class following the agent pattern.
- Use Pydantic for output models.
- Integrate with `app.py` by adding sidebar controls and tab outputs.

## References
- See `diagnosis.py` and `schemes.py` for agent structure and prompt engineering.
- See `crop-recommendation-rf/README.md` for ML workflow details.

---
For questions or unclear conventions, review agent class docstrings and Streamlit UI logic in `app.py`.
