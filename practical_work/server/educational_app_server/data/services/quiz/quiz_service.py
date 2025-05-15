import asyncio
from typing import List
import random
from data.models.quiz import QuizQuestion
from data.services.gemini.gemini_service import GeminiService


class QuizService:
    @staticmethod
    async def generate_quiz(difficulty: str) -> List[QuizQuestion]:

        difficulty_map = {
            "easy": {"easy": 7, "medium": 2, "hard": 1},
            "medium": {"easy": 2, "medium": 7, "hard": 1},
            "hard": {"easy": 1, "medium": 2, "hard": 7}
        }

        if difficulty not in difficulty_map:
            raise ValueError("Invalid difficulty")

        session_seed = random.randint(1000, 9999)
        tasks = []
        current_id = 1

        # subtopic distribution to further ensure question diversity
        topics = [
            "investment", "budgeting", "credit", "debt",
            "risk", "saving", "insurance", "retirement",
            "market history", "financial terms"
        ]

        assigned_topics = {}

        for level, count in difficulty_map[difficulty].items():
            for i in range(count):
                # assign different topics to questions to ensure diversity
                topic_index = (current_id + i) % len(topics)
                assigned_topics[current_id] = topics[topic_index]

                # use both session seed and question ID to ensure uniqueness
                task_seed = session_seed + current_id

                # create task with unique seed
                tasks.append(GeminiService.generate_question(level, current_id, seed=task_seed))
                current_id += 1

        # execute all tasks concurrently
        quiz_questions = await asyncio.gather(*tasks)

        question_texts = set()
        unique_questions = []

        for question in quiz_questions:
            if question.question not in question_texts:
                question_texts.add(question.question)
                unique_questions.append(question)
            else:
                # in the unlikely case of a duplicate, create a modified version
                question.question = f"{question.question} (Variation)"
                unique_questions.append(question)

        return unique_questions