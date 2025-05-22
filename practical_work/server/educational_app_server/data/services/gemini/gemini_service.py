# import json
# import aiohttp
# import random
# from data.models.quiz import QuizQuestion, QuizAnswer
# from exceptions.QuotaExceededException import QuotaExceededException
#
# GEMINI_API_KEY = "AIzaSyBxco91enT-HdwfLb8KoBeG-YhMe_SX2iM"
#
# # cache of generated questions to avoid duplicates
# # question_hashes: Set[str] = set()
#
#
# class GeminiService:
#     API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
#
#     CATEGORIES = [
#         "investment strategies",
#         "budgeting techniques",
#         "credit management",
#         "debt reduction",
#         "risk management",
#         "saving plans",
#         "insurance policies",
#         "retirement planning",
#         "market crashes",
#         "economic bubbles",
#         "financial terms",
#         "personal finance",
#         "tax planning",
#         "estate planning",
#         "financial ratios",
#         "accounting principles"
#     ]
#
#     @staticmethod
#     async def generate_question(difficulty: str, index: int, seed: int = None) -> QuizQuestion:
#         # api_key = os.environ.get("GEMINI_API_KEY")
#         api_key = GEMINI_API_KEY
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY environment variable not set")
#
#         if seed is None:
#             seed = index
#
#         random.seed(seed + index)
#         selected_category = random.choice(GeminiService.CATEGORIES)
#
#         prompt = (
#             f"Generate a {difficulty} financial literacy multiple choice question with 4 options specifically about "
#             f"{selected_category}.\n\n"
#             f"The question should be unique and focus on testing practical knowledge that is valuable for financial literacy.\n\n"
#             f"Format your response as a JSON object with the following structure:\n"
#             f"{{\n"
#             f'  "question": "The question text here",\n'
#             f'  "category": "One of: investing, saving, credit, debt, budgeting, insurance, retirement, history, terms",\n'
#             f'  "answers": [\n'
#             f'    {{"text": "First answer option", "isCorrect": true or false}},\n'
#             f'    {{"text": "Second answer option", "isCorrect": true or false}},\n'
#             f'    {{"text": "Third answer option", "isCorrect": true or false}},\n'
#             f'    {{"text": "Fourth answer option", "isCorrect": true or false}}\n'
#             f'  ]\n'
#             f"}}\n\n"
#             f"IMPORTANT: Make sure exactly one answer is marked as correct (isCorrect: true). "
#             f"The question ID is {index} and difficulty is {difficulty}. Make this question distinct from others."
#         )
#
#         headers = {"Content-Type": "application/json"}
#         payload = {
#             "contents": [
#                 {
#                     "role": "user",
#                     "parts": [{"text": prompt}]
#                 }
#             ],
#             # randomness to ensure different outputs
#             "generationConfig": {
#                 "temperature": 0.7 + (index * 0.05) % 0.3,  # between 0.7 and 1.0
#                 "topP": 0.8,
#                 "topK": 40
#             }
#         }
#
#         url = f"{GeminiService.API_URL}?key={api_key}"
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, headers=headers, json=payload) as response:
#                 if response.status == 429:
#                     error_data = await response.json()
#
#                     retry_delay = 60  # default
#                     retry_info = next(
#                         (item for item in error_data.get("error", {}).get("details", [])
#                          if item.get("@type", "").endswith("RetryInfo")),
#                         None
#                     )
#                     if retry_info and "retryDelay" in retry_info:
#                         retry_str = retry_info["retryDelay"].replace("s", "")
#                         try:
#                             retry_delay = int(float(retry_str))
#                         except ValueError:
#                             pass
#
#                     raise QuotaExceededException(retry_delay,
#                                                  message=error_data["error"].get("message", "Quota exceeded"))
#
#                 elif response.status != 200:
#                     error_text = await response.text()
#                     raise Exception(f"Gemini API error: {response.status} - {error_text}")
#
#                 data = await response.json()
#                 response_text = data["candidates"][0]["content"]["parts"][0]["text"]
#
#                 try:
#                     if "```json" in response_text:
#                         json_text = response_text.split("```json")[1].split("```")[0].strip()
#                     elif "```" in response_text:
#                         json_text = response_text.split("```")[1].strip()
#                     else:
#                         json_text = response_text.strip()
#
#                     data = json.loads(json_text)
#
#                     question_text = data["question"]
#                     category = data.get("category", "general")
#
#                     # Check if we've seen a similar question before
#                     # question_hash = hashlib.md5(question_text.lower().encode()).hexdigest()
#
#                     # if question_hash in question_hashes:
#                     #     question_text = f"{question_text} (Variant {index})"
#                     #
#                     # question_hashes.add(question_hash)
#
#                     quiz_answers = []
#                     for answer in data["answers"]:
#                         quiz_answers.append(
#                             QuizAnswer(
#                                 text=answer["text"],
#                                 isCorrect=answer["isCorrect"]
#                             )
#                         )
#
#                     return QuizQuestion(
#                         id=index,
#                         question=question_text,
#                         difficulty=difficulty,
#                         category=category,
#                         allAnswers=quiz_answers
#                     )
#
#                 except (json.JSONDecodeError, KeyError) as e:
#                     print(f"Error parsing API response: {e}")
#                     print(f"Raw response: {response_text}")
#
#                     # generate a more specific fallback question using the index for variety
#                     topics = ["investing", "budgeting", "saving", "credit", "taxes", "retirement"]
#                     fallback_topic = topics[index % len(topics)]
#
#                     answers = [
#                         QuizAnswer(text=f"Correct {fallback_topic} answer", isCorrect=True),
#                         QuizAnswer(text=f"Wrong {fallback_topic} answer A", isCorrect=False),
#                         QuizAnswer(text=f"Wrong {fallback_topic} answer B", isCorrect=False),
#                         QuizAnswer(text=f"Wrong {fallback_topic} answer C", isCorrect=False),
#                     ]
#                     return QuizQuestion(
#                         id=index,
#                         question=f"What is a key {difficulty} concept in {fallback_topic}?",
#                         difficulty=difficulty,
#                         category=fallback_topic,
#                         allAnswers=answers
#                     )
#
#     @staticmethod
#     def generate_fallback_question(index: int, difficulty: str) -> QuizQuestion:
#         fallback_questions = [
#             {
#                 "question": "What is a common benefit of budgeting?",
#                 "category": "budgeting",
#                 "answers": [
#                     {"text": "It helps track income and expenses", "isCorrect": True},
#                     {"text": "It increases your credit score", "isCorrect": False},
#                     {"text": "It eliminates taxes", "isCorrect": False},
#                     {"text": "It guarantees high investment returns", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "Which of the following is considered a good saving habit?",
#                 "category": "saving",
#                 "answers": [
#                     {"text": "Setting aside money every month", "isCorrect": True},
#                     {"text": "Spending all income", "isCorrect": False},
#                     {"text": "Using payday loans", "isCorrect": False},
#                     {"text": "Keeping cash at home", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "What does a credit score represent?",
#                 "category": "credit",
#                 "answers": [
#                     {"text": "Your creditworthiness to lenders", "isCorrect": True},
#                     {"text": "Your monthly income", "isCorrect": False},
#                     {"text": "Your savings balance", "isCorrect": False},
#                     {"text": "Your tax bracket", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "Which of the following is a long-term investment?",
#                 "category": "investment",
#                 "answers": [
#                     {"text": "Buying government bonds", "isCorrect": True},
#                     {"text": "Daily grocery shopping", "isCorrect": False},
#                     {"text": "Paying monthly rent", "isCorrect": False},
#                     {"text": "Dining at a restaurant", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "Why is diversification important in investing?",
#                 "category": "investment",
#                 "answers": [
#                     {"text": "It reduces overall risk", "isCorrect": True},
#                     {"text": "It guarantees returns", "isCorrect": False},
#                     {"text": "It avoids taxes", "isCorrect": False},
#                     {"text": "It maximizes short-term gains", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "What is a common feature of a high-interest savings account?",
#                 "category": "saving",
#                 "answers": [
#                     {"text": "Higher return on saved money", "isCorrect": True},
#                     {"text": "Unlimited withdrawals", "isCorrect": False},
#                     {"text": "No bank fees", "isCorrect": False},
#                     {"text": "It acts as a credit card", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "What does insurance help with?",
#                 "category": "insurance",
#                 "answers": [
#                     {"text": "Protects against financial loss", "isCorrect": True},
#                     {"text": "Increases your monthly income", "isCorrect": False},
#                     {"text": "Avoids budgeting", "isCorrect": False},
#                     {"text": "Eliminates debt", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "What is a common retirement plan?",
#                 "category": "retirement",
#                 "answers": [
#                     {"text": "401(k)", "isCorrect": True},
#                     {"text": "Car loan", "isCorrect": False},
#                     {"text": "Credit card", "isCorrect": False},
#                     {"text": "Savings bond", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "What is one risk of having too much debt?",
#                 "category": "debt",
#                 "answers": [
#                     {"text": "Difficulty getting new credit", "isCorrect": True},
#                     {"text": "Lower taxes", "isCorrect": False},
#                     {"text": "Higher savings", "isCorrect": False},
#                     {"text": "More insurance coverage", "isCorrect": False}
#                 ]
#             },
#             {
#                 "question": "Which term refers to ownership in a company?",
#                 "category": "financial terms",
#                 "answers": [
#                     {"text": "Stock", "isCorrect": True},
#                     {"text": "Loan", "isCorrect": False},
#                     {"text": "Mortgage", "isCorrect": False},
#                     {"text": "Credit", "isCorrect": False}
#                 ]
#             }
#         ]
#
#         data = fallback_questions[index % len(fallback_questions)]
#
#         quiz_answers = [
#             QuizAnswer(text=a["text"], isCorrect=a["isCorrect"])
#             for a in data["answers"]
#         ]
#
#         return QuizQuestion(
#             id=index,
#             question=data["question"],
#             difficulty=difficulty,
#             category=data["category"],
#             allAnswers=quiz_answers
#         )







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
                "temperature": 0.9,
                "topP": 0.8,
                "topK": 40
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
