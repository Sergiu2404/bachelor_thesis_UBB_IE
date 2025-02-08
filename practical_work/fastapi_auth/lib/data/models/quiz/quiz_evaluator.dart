import 'package:fastapi_auth/data/models/quiz/quiz_question.dart';


// RULES
// - Fully correct → Full points (1.0)
// - If incorrect answers are chosen → No points (0.0)
// - If some correct are chosen but some missed → Partial score
class QuizEvaluator {
  // eval score based on selected answers
  static double evaluateScore(QuizQuestion question, List<int> selectedIndices) {
    List<int> correctIndices = question.getCorrectAnswerIndices();
    List<int> incorrectIndices = selectedIndices
        .where((index) => !correctIndices.contains(index))
        .toList();

    int totalCorrect = correctIndices.length;
    int correctlySelected = selectedIndices.where((index) => correctIndices.contains(index)).length;
    int incorrectSelected = incorrectIndices.length;


    if (incorrectSelected > 0) {
      return 0.0; // Any incorrect answer = No points
    }

    if (correctlySelected == totalCorrect) {
      return 1.0; // Fully correct = Full points
    }

    // Partial scoring formula:
    return correctlySelected / totalCorrect;
  }
}
