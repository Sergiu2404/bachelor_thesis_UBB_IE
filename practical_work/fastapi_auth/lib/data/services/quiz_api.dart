import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/quiz/quiz_question.dart';
import 'auth_api.dart';


class QuizService {
  static const String baseUrl = 'http://10.0.2.2:8000/quiz';

  // Future<List<QuizQuestion>> getQuiz(String difficulty) async {
  //   final response = await http.get(Uri.parse('$baseUrl/$difficulty'));
  //
  //   if (response.statusCode == 200) {
  //     List<dynamic> data = jsonDecode(response.body);
  //     return data.map((q) => QuizQuestion.fromJson(q)).toList();
  //   } else {
  //     throw Exception('Failed to load quiz questions');
  //   }
  // }

  Future<List<QuizQuestion>> getQuiz(String difficulty) async {
    final token = await AuthService().getToken();

    if (token == null) {
      throw Exception('User is not authenticated.');
    }

    final response = await http.get(
      Uri.parse('$baseUrl/$difficulty'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
      },
    );

    if (response.statusCode == 200) {
      List<dynamic> data = jsonDecode(response.body);
      return data.map((q) => QuizQuestion.fromJson(q)).toList();
    } else if (response.statusCode == 401) {
      throw Exception('Unauthorized: Please log in again.');
    } else {
      throw Exception('Failed to load quiz questions (${response.statusCode})');
    }
  }
}
