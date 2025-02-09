import 'package:flutter/material.dart';
import '../../../data/models/quiz/quiz_question.dart';
import '../../../data/models/quiz/quiz_evaluator.dart';
import '../../../data/repositories/quiz_repository.dart';
import '../../../data/services/auth_api.dart';

const QUESTIONS_NR = 3;

class QuizQuestionPage extends StatefulWidget {
  final String difficulty;
  final int reward;

  const QuizQuestionPage({
    super.key,
    required this.difficulty,
    required this.reward,
  });

  @override
  State<QuizQuestionPage> createState() => _QuizQuestionPageState();
}

class _QuizQuestionPageState extends State<QuizQuestionPage> {
  late List<QuizQuestion> allQuestions;
  late List<QuizQuestion> selectedQuestions;

  Map<int, Set<int>> userAnswers = {};

  int currentQuestionIndex = 0;
  double totalScore = 0;
  Set<int> selectedAnswers = {};
  bool canProceed = false;

  // @override
  // void initState() {
  //   super.initState();
  //   // Get all questions and filter by difficulty
  //   allQuestions = QuizRepository.getFinanceQuizQuestions()
  //       .where((q) => q.difficulty == widget.difficulty)
  //       .toList();
  //
  //   // Randomly select 15 questions
  //   allQuestions.shuffle();
  //   selectedQuestions = allQuestions.take(QUESTIONS_NR).toList();
  // }

  // void _handleAnswerSelection(int answerIndex, bool? value) {
  //   setState(() {
  //     if (value == true) {
  //       selectedAnswers.add(answerIndex);
  //     } else {
  //       selectedAnswers.remove(answerIndex);
  //     }
  //     canProceed = selectedAnswers.isNotEmpty;
  //   });
  // }

  void _handleAnswerSelection(int answerIndex, bool? value) {
    setState(() {
      if (value == true) {
        userAnswers.putIfAbsent(currentQuestionIndex, () => {}).add(answerIndex);
        selectedAnswers.add(answerIndex); // Update both state variables
      } else {
        userAnswers[currentQuestionIndex]?.remove(answerIndex);
        selectedAnswers.remove(answerIndex); // Update both state variables
      }
      canProceed = userAnswers[currentQuestionIndex]?.isNotEmpty ?? false;
    });
  }

// Also update your initState to initialize selectedAnswers
  @override
  void initState() {
    super.initState();
    // Get all questions and filter by difficulty
    allQuestions = QuizRepository.getFinanceQuizQuestions()
        .where((q) => q.difficulty == widget.difficulty)
        .toList();

    // Randomly select questions
    allQuestions.shuffle();
    selectedQuestions = allQuestions.take(QUESTIONS_NR).toList();

    // Initialize selectedAnswers with any existing answers for the first question
    selectedAnswers = userAnswers[0] ?? {};
  }


  // void _nextQuestion() {
  //   if (!canProceed) return;
  //
  //   // Evaluate current question score
  //   double questionScore = QuizEvaluator.evaluateScore(
  //     selectedQuestions[currentQuestionIndex],
  //     selectedAnswers.toList(),
  //   );
  //
  //   totalScore += questionScore;
  //
  //   if (currentQuestionIndex < selectedQuestions.length - 1) {
  //     setState(() {
  //       currentQuestionIndex++;
  //       selectedAnswers.clear();
  //       canProceed = false;
  //     });
  //   } else {
  //     _showResult();
  //   }
  // }
  void _nextQuestion() {
    if (!canProceed) return;

    double questionScore = QuizEvaluator.evaluateScore(
      selectedQuestions[currentQuestionIndex],
      userAnswers[currentQuestionIndex]?.toList() ?? [],
    );
    totalScore += questionScore;

    if (currentQuestionIndex < selectedQuestions.length - 1) {
      setState(() {
        currentQuestionIndex++;
        selectedAnswers = userAnswers[currentQuestionIndex] ?? {}; // Restore answers
        canProceed = selectedAnswers.isNotEmpty;
      });
    } else {
      _showResult();
    }
  }



  void _showResult() async {
    final int correctAnswers = totalScore.round();
    final bool passed = correctAnswers >= 1;

    if (passed) {
      final user = await AuthService().getCurrentUser();
      double newBalance = user["virtual_money_balance"] + widget.reward;
      await AuthService().updateUserBalance(newBalance);
    }

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        title: Text(
          passed ? "Congratulations! 🎉" : "Keep Learning! 📚",
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
            color: passed ? Colors.green : Colors.orange,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              "You got $correctAnswers out of $QUESTIONS_NR questions correct!",
              style: TextStyle(fontSize: 18),
            ),
            SizedBox(height: 16),
            Text(
              passed
                  ? "You've earned \$${widget.reward}! 💰"
                  : "Try again to earn \$${widget.reward}!",
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              // First, close the dialog
              Navigator.of(context).pop();
              // Then navigate to quiz page using named route
              Navigator.of(context).pushNamedAndRemoveUntil(
                '/quiz-page',
                    (route) => false, // This removes all routes from the stack
              );
            },
            child: Text("Back to Quiz Menu"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final currentQuestion = selectedQuestions[currentQuestionIndex];

    return Scaffold(
      appBar: AppBar(
        title: Text("Question ${currentQuestionIndex + 1}/$QUESTIONS_NR"),
        leading: currentQuestionIndex > 0
            ? IconButton(
          icon: Icon(Icons.arrow_back),
          onPressed: () {
            setState(() {
              currentQuestionIndex--;
              selectedAnswers = userAnswers[currentQuestionIndex] ?? {}; // Restore answers
              canProceed = selectedAnswers.isNotEmpty;
            });
          },
        )
            : null,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Progress indicator
            LinearProgressIndicator(
              value: (currentQuestionIndex + 1) / QUESTIONS_NR,
              backgroundColor: Colors.grey[200],
              valueColor: AlwaysStoppedAnimation<Color>(Colors.blue),
            ),
            SizedBox(height: 24),

            // Question
            Text(
              currentQuestion.question,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 24),

            // Answers
            Expanded(
              child: ListView.builder(
                itemCount: currentQuestion.allAnswers.length,
                itemBuilder: (context, index) {
                  final answer = currentQuestion.allAnswers[index];
                  return Card(
                    elevation: 2,
                    margin: EdgeInsets.symmetric(vertical: 8),
                    child: CheckboxListTile(
                      title: Text(
                        answer.text ?? "",
                        style: TextStyle(fontSize: 16),
                      ),
                      value: selectedAnswers.contains(index),
                      onChanged: (bool? value) {
                        _handleAnswerSelection(index, value);
                      },
                    ),
                  );
                },
              ),
            ),

            // Next button
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: canProceed ? _nextQuestion : null,
              style: ElevatedButton.styleFrom(
                padding: EdgeInsets.symmetric(vertical: 16),
                backgroundColor: Colors.blue,
                disabledBackgroundColor: Colors.grey,
              ),
              child: Text(
                currentQuestionIndex < selectedQuestions.length - 1
                    ? "Next Question"
                    : "Finish Quiz",
                style: TextStyle(
                  fontSize: 18,
                  color: Colors.white,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}