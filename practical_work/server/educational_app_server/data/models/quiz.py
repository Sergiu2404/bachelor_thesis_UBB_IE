from pydantic import BaseModel
from typing import List

class QuizAnswer(BaseModel):
    text: str
    isCorrect: bool

class QuizQuestion(BaseModel):
    id: int
    question: str
    difficulty: str
    category: str
    allAnswers: List[QuizAnswer]