import 'package:flutter/material.dart';
import '../../../data/models/quiz/quiz_question.dart';
import '../../../data/services/quiz_api.dart';
import '../../../data/services/auth_api.dart';

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
  final QuizService _quizService = QuizService();
  final AuthService _authService = AuthService();

  List<QuizQuestion> _questions = [];
  int _currentIndex = 0;
  bool _isLoading = true;

  // Stores selected answers: questionId -> Set of selected indices
  Map<int, Set<int>> _selectedAnswers = {};

  @override
  void initState() {
    super.initState();
    _loadQuestions();
  }

  Future<void> _loadQuestions() async {
    try {
      List<QuizQuestion> questions =
      await _quizService.getQuiz(widget.difficulty);
      setState(() {
        _questions = questions.take(10).toList(); // Limit to 10 questions
        _isLoading = false;
      });
    } catch (e) {
      print('Failed to load questions: $e');
    }
  }

  void _toggleAnswer(int index, bool isSelected) {
    final questionId = _questions[_currentIndex].id;
    setState(() {
      _selectedAnswers.putIfAbsent(questionId, () => {});
      if (isSelected) {
        _selectedAnswers[questionId]!.add(index);
      } else {
        _selectedAnswers[questionId]!.remove(index);
      }
    });
  }

  void _nextQuestion() {
    if (_currentIndex < _questions.length - 1) {
      setState(() {
        _currentIndex++;
      });
    }
  }

  void _previousQuestion() {
    if (_currentIndex > 0) {
      setState(() {
        _currentIndex--;
      });
    }
  }

  void _finishQuiz() async {
    int correctCount = 0;

    for (var q in _questions) {
      final selected = _selectedAnswers[q.id] ?? {};
      final correct = <int>[];

      for (int i = 0; i < q.allAnswers.length; i++) {
        if (q.allAnswers[i].isCorrect == true) {
          correct.add(i);
        }
      }

      if (selected.length == correct.length && selected.every(correct.contains)) {
        correctCount++;
      }
    }

    // Update balance if passed
    if (correctCount >= 9) {
      final user = await _authService.getCurrentUser();
      double currentBalance = user["virtual_money_balance"] ?? 0;
      double newBalance = currentBalance + widget.reward;
      await _authService.updateUserBalance(newBalance);
    }

    // Show result dialog
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        title: Text(
          correctCount >= 9 ? "Congratulations! 🎉" : "Keep Learning! 📚",
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: correctCount >= 9 ? Colors.green : Colors.orange,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text("You got $correctCount out of 10 questions correct."),
            const SizedBox(height: 10),
            Text(
              correctCount >= 9
                  ? "You've earned \$${widget.reward}! 💰"
                  : "Try again to earn \$${widget.reward}.",
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              Navigator.of(context).pushNamedAndRemoveUntil(
                '/quiz-page',
                    (route) => false,
              );
            },
            child: const Text("Back to Quiz Menu"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final question = _questions[_currentIndex];
    final selected = _selectedAnswers[question.id] ?? {};

    return Scaffold(
      appBar: AppBar(
        title: Text('Question ${_currentIndex + 1}/${_questions.length}'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              question.question,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            Expanded(
              child: ListView.builder(
                itemCount: question.allAnswers.length,
                itemBuilder: (context, i) {
                  final answer = question.allAnswers[i];
                  return CheckboxListTile(
                    value: selected.contains(i),
                    onChanged: (bool? value) {
                      _toggleAnswer(i, value ?? false);
                    },
                    title: Text(answer.text),
                  );
                },
              ),
            ),
            const SizedBox(height: 16),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                if (_currentIndex > 0)
                  ElevatedButton(
                    onPressed: _previousQuestion,
                    child: const Text('Previous'),
                  ),
                if (_currentIndex < _questions.length - 1)
                  ElevatedButton(
                    onPressed: _nextQuestion,
                    child: const Text('Next'),
                  )
                else
                  ElevatedButton(
                    onPressed: _finishQuiz,
                    child: const Text('Finish'),
                  ),
              ],
            )
          ],
        ),
      ),
    );
  }
}
