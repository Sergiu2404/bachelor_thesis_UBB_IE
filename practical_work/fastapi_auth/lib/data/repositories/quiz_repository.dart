import 'package:fastapi_auth/data/models/quiz/quiz_answer.dart';
import 'package:fastapi_auth/data/models/quiz/quiz_question.dart';

class QuizRepository {
  static List<QuizQuestion> getFinanceQuizQuestions() {
    return [
      QuizQuestion(
        id: 1,
        question: "What is the primary goal of investing?",
        difficulty: "easy",
        category: "investing",
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
        category: "investing",
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
        category: "investing",
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
        category: "investing",
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
        category: "investing",
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
        category: "investing",
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
        category: "economy",
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
        category: "investing",
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
        category: "markets",
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
        category: "investing",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 10, text: "Dividends from stocks", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 10, text: "Salary from a job", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 10, text: "Freelance consulting", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 10, text: "Winning a lottery", isCorrect: false),
        ],
      ),








      QuizQuestion(
        id: 11,
        question: "What major event triggered the Great Depression in 1929?",
        difficulty: "hard",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 11, text: "The Stock Market Crash", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 11, text: "The End of World War I", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 11, text: "The Collapse of the Bretton Woods System", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 11, text: "The Rise of the Soviet Union", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 12,
        question: "Which financial instrument is used by governments to raise capital?",
        difficulty: "medium",
        category: "markets",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 12, text: "Bonds", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 12, text: "Stocks", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 12, text: "Real Estate", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 12, text: "Commodities", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 13,
        question: "Which company was the first to reach a market capitalization of \$1 trillion?",
        difficulty: "hard",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 13, text: "Apple", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 13, text: "Microsoft", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 13, text: "Amazon", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 13, text: "Alphabet (Google)", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 14,
        question: "What is the primary function of the Federal Reserve?",
        difficulty: "medium",
        category: "markets",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 14, text: "To control monetary policy and stabilize the economy", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 14, text: "To regulate the stock market", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 14, text: "To supervise the banking industry", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 14, text: "To issue stock for the U.S. government", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 15,
        question: "What is the term for the total value of goods and services produced by a country?",
        difficulty: "easy",
        category: "economy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 15, text: "Gross Domestic Product (GDP)", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 15, text: "Consumer Price Index (CPI)", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 15, text: "Inflation Rate", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 15, text: "Interest Rate", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 16,
        question: "Which event in 2008 led to a global financial crisis?",
        difficulty: "medium",
        category: "investing",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 16, text: "The Collapse of Lehman Brothers", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 16, text: "The Dot-com Bubble Burst", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 16, text: "The Brexit Vote", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 16, text: "The Oil Crisis", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 17,
        question: "What does the 'Dow Jones Industrial Average' represent?",
        difficulty: "medium",
        category: "markets",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 17, text: "A stock market index of 30 major U.S. companies", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 17, text: "The total value of U.S. government bonds", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 17, text: "A measure of inflation in the U.S.", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 17, text: "A measure of U.S. GDP growth", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 18,
        question: "What is the significance of the Bretton Woods Agreement of 1944?",
        difficulty: "hard",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 18, text: "It established the U.S. dollar as the world's reserve currency", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 18, text: "It created the European Union", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 18, text: "It created the World Trade Organization", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 18, text: "It introduced the gold standard", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 19,
        question: "In which year did the stock market crash known as 'Black Monday' occur?",
        difficulty: "medium",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 19, text: "1987", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 19, text: "2001", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 19, text: "1929", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 19, text: "2008", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 20,
        question: "What does the term 'stagflation' refer to?",
        difficulty: "hard",
        category: "economy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 20, text: "A period of high inflation and high unemployment", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 20, text: "A period of low inflation and low unemployment", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 20, text: "A period of high economic growth", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 20, text: "A period of deflation and high unemployment", isCorrect: false),
        ],
      ),






      QuizQuestion(
        id: 21,
        question: "What is the 'S&P 500' index?",
        difficulty: "medium",
        category: "markets",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 21, text: "A stock market index of 500 major U.S. companies", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 21, text: "A measure of U.S. GDP growth", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 21, text: "A government bond index", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 21, text: "The average price of commodities", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 22,
        question: "Which financial crisis was triggered by a housing bubble in 2008?",
        difficulty: "medium",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 22, text: "The Global Financial Crisis", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 22, text: "The Dot-com Bubble", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 22, text: "The Asian Financial Crisis", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 22, text: "The 2001 Recession", isCorrect: false),
        ],
      ),

      QuizQuestion(
        id: 23,
        question: "What is an 'exchange-traded fund' (ETF)?",
        difficulty: "easy",
        category: "investing",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 23, text: "A fund traded on stock exchanges, much like stocks", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 23, text: "A government bond investment", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 23, text: "A type of savings account", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 23, text: "A type of currency exchange product", isCorrect: false),
        ],
      ),
      QuizQuestion(
        id: 24,
        question: "Which company introduced the first initial public offering (IPO) in the U.S. stock market?",
        difficulty: "medium",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 24, text: "Bank of New York", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 24, text: "Apple", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 24, text: "Microsoft", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 24, text: "Amazon", isCorrect: false),
        ],
      ),
      QuizQuestion(
        id: 25,
        question: "What is a 'blue-chip stock'?",
        difficulty: "easy",
        category: "investing",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 25, text: "A stock from a well-established and financially sound company", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 25, text: "A penny stock", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 25, text: "A government bond", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 25, text: "A cryptocurrency investment", isCorrect: false),
        ],
      ),
      QuizQuestion(
        id: 26,
        question: "What is the 'Nikkei 225' index?",
        difficulty: "medium",
        category: "markets",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 26, text: "A stock market index of Japan's 225 largest companies", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 26, text: "A bond market index", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 26, text: "A measure of Japan's inflation", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 26, text: "A commodity price index", isCorrect: false),
        ],
      ),
      QuizQuestion(
        id: 27,
        question: "Which year did the United States officially abandon the gold standard?",
        difficulty: "hard",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 27, text: "1971", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 27, text: "1929", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 27, text: "1965", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 27, text: "2000", isCorrect: false),
        ],
      ),
      QuizQuestion(
        id: 28,
        question: "What is 'capital gains tax'?",
        difficulty: "easy",
        category: "economy",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 28, text: "Tax on profits from the sale of assets or investments", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 28, text: "Tax on income earned through salary", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 28, text: "Tax on purchases of goods and services", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 28, text: "Tax on dividends from stocks", isCorrect: false),
        ],
      ),
      QuizQuestion(
        id: 29,
        question: "What does the term 'bull market' refer to?",
        difficulty: "medium",
        category: "markets",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 29, text: "A market in which prices are rising or are expected to rise", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 29, text: "A market with declining prices", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 29, text: "A stagnant market with no significant price movement", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 29, text: "A market characterized by high inflation", isCorrect: false),
        ],
      ),
      QuizQuestion(
        id: 30,
        question: "What was the primary cause of the 2008 financial crisis?",
        difficulty: "medium",
        category: "history",
        allAnswers: [
          QuizAnswer.text(quizQuestionId: 30, text: "Subprime mortgage lending and financial derivatives", isCorrect: true),
          QuizAnswer.text(quizQuestionId: 30, text: "The rise of cryptocurrency", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 30, text: "A collapse in global oil prices", isCorrect: false),
          QuizAnswer.text(quizQuestionId: 30, text: "A government shutdown", isCorrect: false),
        ],
      )

    ];
  }
}
