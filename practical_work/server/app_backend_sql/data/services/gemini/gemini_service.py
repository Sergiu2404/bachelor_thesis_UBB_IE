import json
import aiohttp
import random
from data.models.quiz import QuizQuestion, QuizAnswer
from exceptions.QuotaExceededException import QuotaExceededException

GEMINI_API_KEY = "AIzaSyBxco91enT-HdwfLb8KoBeG-YhMe_SX2iM"

class GeminiService:
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    CATEGORIES = [
        "investment strategies", "budgeting techniques", "credit management", "debt reduction",
        "risk management", "saving plans", "insurance policies", "retirement planning",
        "market crashes", "economic bubbles", "financial terms", "personal finance",
        "tax planning", "estate planning", "financial ratios", "accounting principles"
    ]

    @staticmethod
    def generate_fallback_question(index: int, difficulty: str) -> QuizQuestion:
        fallback_questions = [
            {
                "question": "What is a common benefit of budgeting?",
                "category": "budgeting",
                "answers": [
                    {"text": "It helps track income and expenses", "isCorrect": True},
                    {"text": "It increases your credit score", "isCorrect": False},
                    {"text": "It eliminates taxes", "isCorrect": False},
                    {"text": "It guarantees high investment returns", "isCorrect": False}
                ]
            },
            {
                "question": "Which of the following is considered a good saving habit?",
                "category": "saving",
                "answers": [
                    {"text": "Setting aside money every month", "isCorrect": True},
                    {"text": "Spending all income", "isCorrect": False},
                    {"text": "Using payday loans", "isCorrect": False},
                    {"text": "Keeping cash at home", "isCorrect": False}
                ]
            },
            {
                "question": "What does a credit score represent?",
                "category": "credit",
                "answers": [
                    {"text": "Your creditworthiness to lenders", "isCorrect": True},
                    {"text": "Your monthly income", "isCorrect": False},
                    {"text": "Your savings balance", "isCorrect": False},
                    {"text": "Your tax bracket", "isCorrect": False}
                ]
            },
            {
                "question": "Which of the following is a long-term investment?",
                "category": "investment",
                "answers": [
                    {"text": "Buying government bonds", "isCorrect": True},
                    {"text": "Daily grocery shopping", "isCorrect": False},
                    {"text": "Paying monthly rent", "isCorrect": False},
                    {"text": "Dining at a restaurant", "isCorrect": False}
                ]
            },
            {
                "question": "Why is diversification important in investing?",
                "category": "investment",
                "answers": [
                    {"text": "It reduces overall risk", "isCorrect": True},
                    {"text": "It guarantees returns", "isCorrect": False},
                    {"text": "It avoids taxes", "isCorrect": False},
                    {"text": "It maximizes short-term gains", "isCorrect": False}
                ]
            },
            {
                "question": "What is a common feature of a high-interest savings account?",
                "category": "saving",
                "answers": [
                    {"text": "Higher return on saved money", "isCorrect": True},
                    {"text": "Unlimited withdrawals", "isCorrect": False},
                    {"text": "No bank fees", "isCorrect": False},
                    {"text": "It acts as a credit card", "isCorrect": False}
                ]
            },
            {
                "question": "What does insurance help with?",
                "category": "insurance",
                "answers": [
                    {"text": "Protects against financial loss", "isCorrect": True},
                    {"text": "Increases your monthly income", "isCorrect": False},
                    {"text": "Avoids budgeting", "isCorrect": False},
                    {"text": "Eliminates debt", "isCorrect": False}
                ]
            },
            {
                "question": "What is a common retirement plan?",
                "category": "retirement",
                "answers": [
                    {"text": "401(k)", "isCorrect": True},
                    {"text": "Car loan", "isCorrect": False},
                    {"text": "Credit card", "isCorrect": False},
                    {"text": "Savings bond", "isCorrect": False}
                ]
            },
            {
                "question": "What is one risk of having too much debt?",
                "category": "debt",
                "answers": [
                    {"text": "Difficulty getting new credit", "isCorrect": True},
                    {"text": "Lower taxes", "isCorrect": False},
                    {"text": "Higher savings", "isCorrect": False},
                    {"text": "More insurance coverage", "isCorrect": False}
                ]
            },
            {
                "question": "Which term refers to ownership in a company?",
                "category": "financial terms",
                "answers": [
                    {"text": "Stock", "isCorrect": True},
                    {"text": "Loan", "isCorrect": False},
                    {"text": "Mortgage", "isCorrect": False},
                    {"text": "Credit", "isCorrect": False}
                ]
            }
        ]

        data = fallback_questions[index % len(fallback_questions)]

        quiz_answers = [
            QuizAnswer(text=a["text"], isCorrect=a["isCorrect"])
            for a in data["answers"]
        ]

        return QuizQuestion(
            id=index,
            question=data["question"],
            difficulty=difficulty,
            category=data["category"],
            allAnswers=quiz_answers
        )

    @staticmethod
    async def generate_question(difficulty: str, index: int, seed: int = None) -> QuizQuestion:
        api_key = GEMINI_API_KEY
        if not api_key:
            return GeminiService.generate_fallback_question(index, difficulty)

        if seed is None:
            seed = index

        random.seed(seed + index)
        selected_category = random.choice(GeminiService.CATEGORIES)

        prompt = (
            f"Generate a {difficulty} financial literacy multiple choice question with 4 options specifically about "
            f"{selected_category}.\n\n"
            f"The question should be unique and focus on testing practical knowledge that is valuable for financial literacy.\n\n"
            f"Format your response as a JSON object with the following structure:\n"
            f"{{\n"
            f'  "question": "The question text here",\n'
            f'  "category": "...",\n'
            f'  "answers": [\n'
            f'    {{"text": "...", "isCorrect": true or false}},\n'
            f'    ...\n'
            f'  ]\n'
            f"}}"
        )

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.9,  #more creative, less predictable
                "topP": 0.8,  #picks from set of words whose cum probab is >= toppp
                "topK": 40  #limits choices to topk most likely tokens at each step
            }
        }

        url = f"{GeminiService.API_URL}?key={api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        raise Exception(f"Gemini API error: {response.status} - {await response.text()}")

                    data = await response.json()
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"]

                    json_text = response_text.strip()
                    if "```json" in json_text:
                        json_text = json_text.split("```json")[1].split("```", 1)[0].strip()
                    elif "```" in json_text:
                        json_text = json_text.split("```", 1)[1].strip()

                    parsed = json.loads(json_text)
                    answers = [QuizAnswer(text=a["text"], isCorrect=a["isCorrect"]) for a in parsed["answers"]]

                    return QuizQuestion(
                        id=index,
                        question=parsed["question"],
                        difficulty=difficulty,
                        category=parsed.get("category", "general"),
                        allAnswers=answers
                    )
        except Exception as e:
            print(f"[GeminiService] Falling back due to error: {e}")
            return GeminiService.generate_fallback_question(index, difficulty)
