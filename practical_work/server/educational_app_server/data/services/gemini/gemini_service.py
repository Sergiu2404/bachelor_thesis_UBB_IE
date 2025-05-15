# import json
# import aiohttp
# from data.models.quiz import QuizQuestion, QuizAnswer
#
# GEMINI_API_KEY="AIzaSyBxco91enT-HdwfLb8KoBeG-YhMe_SX2iM"
#
# class GeminiService:
#     API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
#
#     @staticmethod
#     async def generate_question(difficulty: str, index: int) -> QuizQuestion:
#         # api_key = os.environ.get("GEMINI_API_KEY")
#         api_key = GEMINI_API_KEY
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY environment variable not set")
#
#         prompt = (
#             f"Generate a {difficulty} financial literacy multiple choice question with 4 options about "
#             f"financial topics like concepts, investment strategies, budgeting, credit, debt, risk management, "
#             f"saving plans, insurance, retirement planning, financial history (market crashes, economic bubbles), "
#             f"or financial terms.\n\n"
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
#             f"IMPORTANT: Make sure exactly one answer is marked as correct (isCorrect: true)."
#         )
#
#         headers = {"Content-Type": "application/json"}
#         payload = {
#             "contents": [
#                 {
#                     "role": "user",
#                     "parts": [{"text": prompt}]
#                 }
#             ]
#         }
#
#         url = f"{GeminiService.API_URL}?key={api_key}"
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, headers=headers, json=payload) as response:
#                 if response.status != 200:
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
#                     answers = [
#                         QuizAnswer(text="Correct answer", isCorrect=True),
#                         QuizAnswer(text="Wrong answer A", isCorrect=False),
#                         QuizAnswer(text="Wrong answer B", isCorrect=False),
#                         QuizAnswer(text="Wrong answer C", isCorrect=False),
#                     ]
#                     return QuizQuestion(
#                         id=index,
#                         question=f"What is a {difficulty} finance concept?",
#                         difficulty=difficulty,
#                         category="investing",
#                         allAnswers=answers
#                     )


import json
import aiohttp
import random
from data.models.quiz import QuizQuestion, QuizAnswer

GEMINI_API_KEY = "AIzaSyBxco91enT-HdwfLb8KoBeG-YhMe_SX2iM"

# cache of generated questions to avoid duplicates
# question_hashes: Set[str] = set()


class GeminiService:
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    CATEGORIES = [
        "investment strategies",
        "budgeting techniques",
        "credit management",
        "debt reduction",
        "risk management",
        "saving plans",
        "insurance policies",
        "retirement planning",
        "market crashes",
        "economic bubbles",
        "financial terms",
        "personal finance",
        "tax planning",
        "estate planning",
        "financial ratios",
        "accounting principles"
    ]

    @staticmethod
    async def generate_question(difficulty: str, index: int, seed: int = None) -> QuizQuestion:
        # api_key = os.environ.get("GEMINI_API_KEY")
        api_key = GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

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
            f'  "category": "One of: investing, saving, credit, debt, budgeting, insurance, retirement, history, terms",\n'
            f'  "answers": [\n'
            f'    {{"text": "First answer option", "isCorrect": true or false}},\n'
            f'    {{"text": "Second answer option", "isCorrect": true or false}},\n'
            f'    {{"text": "Third answer option", "isCorrect": true or false}},\n'
            f'    {{"text": "Fourth answer option", "isCorrect": true or false}}\n'
            f'  ]\n'
            f"}}\n\n"
            f"IMPORTANT: Make sure exactly one answer is marked as correct (isCorrect: true). "
            f"The question ID is {index} and difficulty is {difficulty}. Make this question distinct from others."
        )

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            # randomness to ensure different outputs
            "generationConfig": {
                "temperature": 0.7 + (index * 0.05) % 0.3,  # between 0.7 and 1.0
                "topP": 0.8,
                "topK": 40
            }
        }

        url = f"{GeminiService.API_URL}?key={api_key}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Gemini API error: {response.status} - {error_text}")

                data = await response.json()
                response_text = data["candidates"][0]["content"]["parts"][0]["text"]

                try:
                    if "```json" in response_text:
                        json_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        json_text = response_text.split("```")[1].strip()
                    else:
                        json_text = response_text.strip()

                    data = json.loads(json_text)

                    question_text = data["question"]
                    category = data.get("category", "general")

                    # Check if we've seen a similar question before
                    # question_hash = hashlib.md5(question_text.lower().encode()).hexdigest()

                    # if question_hash in question_hashes:
                    #     question_text = f"{question_text} (Variant {index})"
                    #
                    # question_hashes.add(question_hash)

                    quiz_answers = []
                    for answer in data["answers"]:
                        quiz_answers.append(
                            QuizAnswer(
                                text=answer["text"],
                                isCorrect=answer["isCorrect"]
                            )
                        )

                    return QuizQuestion(
                        id=index,
                        question=question_text,
                        difficulty=difficulty,
                        category=category,
                        allAnswers=quiz_answers
                    )

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error parsing API response: {e}")
                    print(f"Raw response: {response_text}")

                    # generate a more specific fallback question using the index for variety
                    topics = ["investing", "budgeting", "saving", "credit", "taxes", "retirement"]
                    fallback_topic = topics[index % len(topics)]

                    answers = [
                        QuizAnswer(text=f"Correct {fallback_topic} answer", isCorrect=True),
                        QuizAnswer(text=f"Wrong {fallback_topic} answer A", isCorrect=False),
                        QuizAnswer(text=f"Wrong {fallback_topic} answer B", isCorrect=False),
                        QuizAnswer(text=f"Wrong {fallback_topic} answer C", isCorrect=False),
                    ]
                    return QuizQuestion(
                        id=index,
                        question=f"What is a key {difficulty} concept in {fallback_topic}?",
                        difficulty=difficulty,
                        category=fallback_topic,
                        allAnswers=answers
                    )