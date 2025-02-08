import 'package:fastapi_auth/data/models/quiz/quiz_answer.dart';

class QuizQuestion {
  int id;
  String question;
  String? photoPath;
  List<QuizAnswer> allAnswers;
  String difficulty; // easy / medium / hard

  QuizQuestion({
    required this.id,
    required this.question,
    this.photoPath,
    required this.allAnswers,
    required this.difficulty,
  });

  List<int> getCorrectAnswerIndices() {
    return allAnswers
        .asMap()
        .entries
        .where((entry) => entry.value.isCorrect)
        .map((entry) => entry.key)
        .toList();
  }

  bool isFullyCorrect(List<int> selectedIndices) {
    List<int> correctIndices = getCorrectAnswerIndices();
    return selectedIndices.toSet().containsAll(correctIndices) &&
        correctIndices.toSet().containsAll(selectedIndices);
  }
}