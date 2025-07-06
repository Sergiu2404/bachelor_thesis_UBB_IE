from starlette.responses import JSONResponse

from data.models.user import User
from data.services.quiz.quiz_service import QuizService
from fastapi import APIRouter, HTTPException, Depends
import traceback

from exceptions.QuotaExceededException import QuotaExceededException
from routers.authentication_router import get_current_user

quiz_router = APIRouter(prefix="/quiz", tags=["quiz"])
@quiz_router.get("/{difficulty}")
async def get_quiz_questions(difficulty: str, current_user: User = Depends(get_current_user)):
    try:
        return await QuizService.generate_quiz(difficulty)
    except ValueError:
        raise HTTPException(status_code=400, detail="Difficulty must be one of: easy, medium, hard")
    except QuotaExceededException as qe:
        return JSONResponse(
            status_code=429,
            content={
                "detail": qe.message,
                "retry_after_seconds": qe.retry_after
            }
        )
    except Exception as e:
        print("Unexpected error in /quiz route:")
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred while generating the quiz."}
        )