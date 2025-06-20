import 'package:fastapi_auth/data/models/quiz/quiz_answer.dart';

// class QuizQuestion {
//   int id;
//   String question;
//   String? photoPath;
//   List<QuizAnswer> allAnswers;
//   String category;
//   String difficulty; // easy / medium / hard
//
//   QuizQuestion({
//     required this.id,
//     required this.question,
//     this.photoPath,
//     required this.allAnswers,
//     required this.category,
//     required this.difficulty,
//   });
//
//   List<int> getCorrectAnswerIndices() {
//     return allAnswers
//         .asMap()
//         .entries
//         .where((entry) => entry.value.isCorrect)
//         .map((entry) => entry.key)
//         .toList();
//   }
//
//   bool isFullyCorrect(List<int> selectedIndices) {
//     List<int> correctIndices = getCorrectAnswerIndices();
//     return selectedIndices.toSet().containsAll(correctIndices) &&
//         correctIndices.toSet().containsAll(selectedIndices);
//   }
// }


class QuizQuestion {
  final int id;
  final String question;
  final String difficulty;
  final String category;
  final List<QuizAnswer> allAnswers;

  QuizQuestion({
    required this.id,
    required this.question,
    required this.difficulty,
    required this.category,
    required this.allAnswers,
  });

  factory QuizQuestion.fromJson(Map<String, dynamic> json) {
    var answersJson = json['allAnswers'] as List;
    List<QuizAnswer> answersList =
    answersJson.map((a) => QuizAnswer.fromJson(a)).toList();

    return QuizQuestion(
      id: json['id'],
      question: json['question'],
      difficulty: json['difficulty'],
      category: json['category'],
      allAnswers: answersList,
    );
  }
}