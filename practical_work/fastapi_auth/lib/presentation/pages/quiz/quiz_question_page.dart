import 'package:flutter/material.dart';

import '../../../data/models/quiz/quiz_question.dart';


class QuizQuestionPage extends StatefulWidget {
  final List<QuizQuestion> questions;

  const QuizQuestionPage({super.key, required this.questions});

  @override
  State<QuizQuestionPage> createState() => _QuizQuestionPageState();
}

class _QuizQuestionPageState extends State<QuizQuestionPage> {
  int currentQuestionIndex = 0;
  int correctAnswers = 0;

  void _nextQuestion(bool isCorrect) {
    if (isCorrect) {
      correctAnswers++;
    }
    if (currentQuestionIndex < widget.questions.length - 1) {
      setState(() {
        currentQuestionIndex++;
      });
    } else {
      _showResult();
    }
  }

  void _showResult() {
    bool passed = correctAnswers >= 14;
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(passed ? "Congratulations!" : "Try Again"),
        content: Text("You answered $correctAnswers out of 15 correctly."),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text("OK"),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final question = widget.questions[currentQuestionIndex];

    return Scaffold(
      appBar: AppBar(title: Text("Question ${currentQuestionIndex + 1}/15")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(question.question, style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            ...question.allAnswers.map((answer) => ListTile(
              title: Text(answer.text ?? ""),
              leading: Radio(
                value: answer.isCorrect,
                groupValue: true,
                onChanged: (value) => _nextQuestion(answer.isCorrect),
              ),
            )),
          ],
        ),
      ),
    );
  }
}
