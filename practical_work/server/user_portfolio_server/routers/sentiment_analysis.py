# import requests
# from fastapi import APIRouter, HTTPException
#
# HUGGINGFACE_URL = "https://hf.space/embed/Sergiu2404/fin_tinybert_space/api/predict/"
# sentiment_analysis_router = APIRouter(prefix="/sentiment_analysis")
#
#
# @sentiment_analysis_router.get("/")
# async def get_sentiment_analysis(text: str):
#     # Sending a POST request to Hugging Face Space API
#     response = requests.post(
#         HUGGINGFACE_URL,
#         headers={"Content-Type": "application/json"},
#         json={"data": [text]}  # Input text is sent as a list in the "data" field
#     )
#
#     if response.status_code == 200:
#         sentiment_result = response.json()
#         sentiment_score, sentiment_interpretation = sentiment_result[0]
#         # sentiment_score = sentiment_result.get("score", None)
#         # sentiment_interpretation = sentiment_result.get("interpretation", None)
#
#         if sentiment_score is None or sentiment_interpretation is None:
#             raise HTTPException(status_code=500, detail="Error retrieving sentiment analysis data")
#
#         return {
#             "sentiment_score": sentiment_score,
#             "interpretation": sentiment_interpretation
#         }
#
#     else:
#         raise HTTPException(status_code=500, detail="Error calling Hugging Face Space API")






# from fastapi import APIRouter, HTTPException
# import requests
#
# # HUGGINGFACE_URL = "https://hf.space/embed/Sergiu2404/fin_tinybert_space/api/predict/"
# HUGGINGFACE_URL = "https://Sergiu2404-fin_tinybert_space.hf.space/api/predict/"
#
# sentiment_analysis_router = APIRouter(prefix="/sentiment_analysis")
#
#
# @sentiment_analysis_router.get("/")
# async def get_sentiment_analysis(text: str):
#     response = requests.post(
#         HUGGINGFACE_URL,
#         headers={"Content-Type": "application/json"},
#         json={"data": [text]}  # Input text is sent as a list in the "data" field
#     )
#
#     if response.status_code == 200:
#         sentiment_result = response.json()
#         sentiment_score, sentiment_interpretation = sentiment_result[0]
#
#         if sentiment_score is None or sentiment_interpretation is None:
#             raise HTTPException(status_code=500, detail="Error retrieving sentiment analysis data")
#
#         return {
#             "sentiment_score": sentiment_score,
#             "interpretation": sentiment_interpretation
#         }
#     else:
#         raise HTTPException(status_code=500, detail="Error calling Hugging Face Space API")



from fastapi import APIRouter, HTTPException
import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The most recent Gradio API format (check the network tab in your browser to confirm)
HUGGINGFACE_URL = "https://Sergiu2404-fin-tinybert-space.hf.space/api/predict"
#HUGGINGFACE_URL = "https://Sergiu2404-fin_tinybert_space.hf.space/run/predict"

# curl -X POST "https://Sergiu2404-fin-tinybert-space.hf.space/api/predict/" \
# -H "Content-Type: application/json" \
# -d "{\"data\": [\"The stock market is performing well.\"]}"

sentiment_analysis_router = APIRouter(prefix="/sentiment_analysis")


@sentiment_analysis_router.get("/")
async def get_sentiment_analysis(text: str):
    try:
        # Format for the latest Gradio API
        payload = {
            "data": [text]
        }

        logger.info(f"Sending request to: {HUGGINGFACE_URL}")
        logger.info(f"Payload: {json.dumps(payload)}")

        response = requests.post(
            HUGGINGFACE_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
            },
            timeout=30
        )

        logger.info(f"Status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")

        # For debugging purposes, try to get raw text
        try:
            logger.info(f"Response content: {response.text[:500]}...")  # First 500 chars
        except Exception as e:
            logger.warning(f"Could not log response text: {str(e)}")

        # Parse JSON response
        if response.status_code == 200:
            try:
                result = response.json()
                logger.info(f"Response structure: {json.dumps(result)[:500]}...")  # First 500 chars

                # Latest Gradio format typically has a 'data' field with the outputs
                if "data" in result and isinstance(result["data"], list) and len(result["data"]) >= 2:
                    sentiment_score = result["data"][0]
                    sentiment_interpretation = result["data"][1]

                    return {
                        "sentiment_score": sentiment_score,
                        "interpretation": sentiment_interpretation
                    }
                else:
                    # Try to extract from different response formats
                    logger.warning(f"Unexpected response format: {json.dumps(result)[:500]}...")

                    # Try to access data directly if it's a different format
                    if isinstance(result, list) and len(result) >= 2:
                        return {
                            "sentiment_score": result[0],
                            "interpretation": result[1]
                        }

                    raise HTTPException(status_code=500,
                                        detail="Could not parse response format from Hugging Face Space API")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {str(e)}")
                raise HTTPException(status_code=500,
                                    detail=f"Failed to parse response from Hugging Face Space API: {str(e)}")
        else:
            error_detail = response.text if response.text else f"Status code: {response.status_code}"
            raise HTTPException(status_code=response.status_code,
                                detail=f"Error calling Hugging Face Space API: {error_detail}")

    except requests.RequestException as e:
        logger.error(f"Request exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request to Hugging Face Space failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")