import asyncio
import time
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import CredibilityRegressor, predict_credibility
from domain_reliability_analyzer import NewsReliabilityChecker
from grammar_analyzer import GrammarAnalyzer
from punctuation_analyzer import PunctuationAnalyzer
import concurrent.futures

app = FastAPI()

grammar_analyzer = GrammarAnalyzer()
punctuation_analyzer = PunctuationAnalyzer()
source_analyzer = NewsReliabilityChecker()

executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

class InputText(BaseModel):
    text: str

def run_grammar_analysis(text: str) -> Dict[str, Any]:
    try:
        start_time = time.time()
        score = grammar_analyzer.run_grammar_analysis(text)
        end_time = time.time()
        return {
            "score": score,
            "execution_time": round(end_time - start_time, 3),
            "error": None
        }
    except Exception as e:
        return {
            "score": 0.5,
            "execution_time": 0,
            "error": str(e)
        }

def run_punctuation_analysis(text: str) -> Dict[str, Any]:
    try:
        start_time = time.time()
        score = punctuation_analyzer.run_punctuation_analysis(text)
        end_time = time.time()
        return {
            "score": score,
            "execution_time": round(end_time - start_time, 3),
            "error": None
        }
    except Exception as e:
        return {
            "score": 0.5,
            "execution_time": 0,
            "error": str(e)
        }

def run_source_analysis(text: str) -> Dict[str, Any]:
    try:
        start_time = time.time()
        score = source_analyzer.get_reliability_score(text)
        end_time = time.time()
        return {
            "score": score,
            "execution_time": round(end_time - start_time, 3),
            "error": None
        }
    except Exception as e:
        return {
            "score": 0.5,
            "execution_time": 0,
            "error": str(e)
        }

def run_ai_model_analysis(text: str) -> Dict[str, Any]:
    try:
        start_time = time.time()
        score = predict_credibility(text)
        end_time = time.time()
        return {
            "score": score,
            "execution_time": round(end_time - start_time, 3),
            "error": None
        }
    except Exception as e:
        return {
            "score": 0.5,
            "execution_time": 0,
            "error": str(e)
        }

# @app.get("/predict_credibility")
# async def predict_credibility_score(input_text: InputText):
#     try:
#         score = predict_credibility(input_text.text)
#         return {
#             "credibility_score": score
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_credibility")
async def predict_credibility_score(input_text: InputText):

    try:
        text = input_text.text

        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        total_start_time = time.time()
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(executor, run_grammar_analysis, text),
            loop.run_in_executor(executor, run_punctuation_analysis, text),
            #loop.run_in_executor(executor, run_source_analysis, text),
            loop.run_in_executor(executor, run_ai_model_analysis, text)
        ]

        results = await asyncio.gather(*tasks)

        total_end_time = time.time()
        total_execution_time = round(total_end_time - total_start_time, 3)

        grammar_result, punctuation_result, ai_result = results

        weights = {
            "grammar": 0.2,
            "punctuation": 0.15,
            "source": 0.25,
            "ai_model": 0.4
        }

        overall_score = (
                grammar_result["score"] * weights["grammar"] +
                punctuation_result["score"] * weights["punctuation"] +
                # source_result["score"] * weights["source"] +
                ai_result["score"] * weights["ai_model"]
        )

        sequential_time = sum([
            grammar_result["execution_time"],
            punctuation_result["execution_time"],
            # source_result["execution_time"],
            ai_result["execution_time"]
        ])

        response = {
            "overall_credibility_score": 0.25 * grammar_result["score"] + 0.25 * punctuation_result["score"] + 0.5 * ai_result["score"],
            "detailed_scores": {
                "grammar_analysis": {
                    "score": grammar_result["score"],
                    "execution_time": grammar_result["execution_time"],
                    "error": grammar_result["error"]
                },
                "punctuation_analysis": {
                    "score": punctuation_result["score"],
                    "execution_time": punctuation_result["execution_time"],
                    "error": punctuation_result["error"]
                },
                # "source_reliability": {
                #     "score": source_result["score"],
                #     "execution_time": source_result["execution_time"],
                #     "error": source_result["error"]
                # },
                "ai_model_analysis": {
                    "score": ai_result["score"],
                    "execution_time": ai_result["execution_time"],
                    "error": ai_result["error"]
                }
            },
            # "performance_metrics": {
            #     "total_parallel_time": total_execution_time,
            #     "estimated_sequential_time": sequential_time,
            #     "time_saved": round(sequential_time - total_execution_time, 3),
            #     "speedup_factor": round(sequential_time / total_execution_time, 2) if total_execution_time > 0 else 1
            # },
            #"weights_used": weights
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")