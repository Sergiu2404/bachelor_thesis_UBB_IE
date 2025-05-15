from data.services.quiz.quiz_service import QuizService
from fastapi import APIRouter, HTTPException
quiz_router = APIRouter(prefix="/quiz", tags=["quiz"])
@quiz_router.get("/{difficulty}")
async def get_quiz_questions(difficulty: str):
    try:
        return await QuizService.generate_quiz(difficulty)
    except ValueError:
        raise HTTPException(status_code=400, detail="Difficulty must be one of: easy, medium, hard")

