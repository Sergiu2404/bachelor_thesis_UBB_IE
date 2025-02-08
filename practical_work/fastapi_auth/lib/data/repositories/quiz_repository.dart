import 'package:fastapi_auth/data/models/quiz/quiz_answer.dart';
import 'package:fastapi_auth/data/models/quiz/quiz_question.dart';

class QuizRepository {
  static List<QuizQuestion> getFinanceQuizQuestions() {
    return [
      QuizQuestion(
        id: 1,
        question: "What is the primary goal of investing?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 1, text: "To generate returns over time", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 1, text: "To store money safely", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 1, text: "To avoid paying taxes", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 1, text: "To minimize spending", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 2,
        question: "Which financial instrument represents ownership in a company?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 2, text: "Stock", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 2, text: "Bond", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 2, text: "Mutual Fund", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 2, text: "Certificate of Deposit (CD)", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 3,
        question: "Which type of investment carries the highest risk?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 3, text: "Stocks", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 3, text: "Savings Account", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 3, text: "Bonds", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 3, text: "Fixed Deposits", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 4,
        question: "What is diversification in investing?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 4, text: "Spreading investments across different assets", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 4, text: "Investing all money in one stock", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 4, text: "Avoiding all risks in the market", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 4, text: "Only investing in government bonds", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 5,
        question: "What does 'liquidity' refer to in finance?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 5, text: "How quickly an asset can be converted into cash", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 5, text: "The interest rate on a loan", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 5, text: "The profit made from an investment", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 5, text: "The level of risk in an investment", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 6,
        question: "What is the safest type of investment?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 6, text: "Government bonds", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 6, text: "Stocks", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 6, text: "Cryptocurrency", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 6, text: "Commodities", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 7,
        question: "Which economic indicator measures inflation?",
        difficulty: "hard",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 7, text: "Consumer Price Index (CPI)", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 7, text: "Gross Domestic Product (GDP)", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 7, text: "Unemployment Rate", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 7, text: "Interest Rate", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 8,
        question: "What is a mutual fund?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 8, text: "A pool of money managed by professionals", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 8, text: "A savings account with a high interest rate", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 8, text: "A type of bond issued by the government", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 8, text: "A direct investment in a single company", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 9,
        question: "Which type of market is associated with rising stock prices?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 9, text: "Bull market", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 9, text: "Bear market", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 9, text: "Stagnant market", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 9, text: "Recession", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 10,
        question: "Which is an example of a passive income source?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 10, text: "Dividends from stocks", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 10, text: "Salary from a job", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 10, text: "Freelance consulting", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 10, text: "Winning a lottery", isCorrect: false),
        ],
      ),









      QuizQuestion(
        id: 11,
        question: "What is the primary goal of investing?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 1, text: "To generate returns over time", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 1, text: "To store money safely", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 1, text: "To avoid paying taxes", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 1, text: "To minimize spending", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 12,
        question: "Which financial instrument represents ownership in a company?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 2, text: "Stock", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 2, text: "Bond", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 2, text: "Mutual Fund", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 2, text: "Certificate of Deposit (CD)", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 13,
        question: "Which type of investment carries the highest risk?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 3, text: "Stocks", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 3, text: "Savings Account", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 3, text: "Bonds", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 3, text: "Fixed Deposits", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 14,
        question: "What is diversification in investing?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 4, text: "Spreading investments across different assets", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 4, text: "Investing all money in one stock", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 4, text: "Avoiding all risks in the market", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 4, text: "Only investing in government bonds", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 15,
        question: "What does 'liquidity' refer to in finance?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 5, text: "How quickly an asset can be converted into cash", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 5, text: "The interest rate on a loan", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 5, text: "The profit made from an investment", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 5, text: "The level of risk in an investment", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 16,
        question: "What is the safest type of investment?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 6, text: "Government bonds", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 6, text: "Stocks", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 6, text: "Cryptocurrency", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 6, text: "Commodities", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 17,
        question: "Which economic indicator measures inflation?",
        difficulty: "hard",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 7, text: "Consumer Price Index (CPI)", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 7, text: "Gross Domestic Product (GDP)", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 7, text: "Unemployment Rate", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 7, text: "Interest Rate", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 18,
        question: "What is a mutual fund?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 8, text: "A pool of money managed by professionals", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 8, text: "A savings account with a high interest rate", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 8, text: "A type of bond issued by the government", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 8, text: "A direct investment in a single company", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 19,
        question: "Which type of market is associated with rising stock prices?",
        difficulty: "easy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 9, text: "Bull market", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 9, text: "Bear market", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 9, text: "Stagnant market", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 9, text: "Recession", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 20,
        question: "Which is an example of a passive income source?",
        difficulty: "medium",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 10, text: "Dividends from stocks", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 10, text: "Salary from a job", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 10, text: "Freelance consulting", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 10, text: "Winning a lottery", isCorrect: false),
        ],
      ),

    ];
  }
}
