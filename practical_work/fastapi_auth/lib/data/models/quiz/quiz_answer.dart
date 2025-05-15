class QuizAnswer {
  int quizQuestionId; // foreign key to question obj (each answer belongs to a question)
  String? text;
  String? photoPath;
  bool isCorrect;

  QuizAnswer.text({
    required this.quizQuestionId,
    required this.text,
    this.photoPath,
    required this.isCorrect,
  });

  QuizAnswer.photo({
    required this.quizQuestionId,
    this.text,
    required this.photoPath,
    required this.isCorrect,
  });

  @override
  String toString() {
    return "$quizQuestionId, ${text ?? "No Text"}, ${photoPath ??
        "No Image"}, $isCorrect\n";
  }
}
//
// class QuizAnswer {
//   final String text;
//   final bool isCorrect;
//
//   QuizAnswer({required this.text, required this.isCorrect});
//
//   factory QuizAnswer.fromJson(Map<String, dynamic> json) {
//     return QuizAnswer(
//       text: json['text'],
//       isCorrect: json['isCorrect'],
//     );
//   }
// }