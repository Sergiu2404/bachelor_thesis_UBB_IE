// // lib/services/auth_service.dart
// import 'dart:convert';
// import 'package:http/http.dart' as http;
// import 'package:shared_preferences/shared_preferences.dart';
//
//
// class QuizService {
//   static const String baseUrl = 'http://10.0.2.2:8000/quiz';
//
//   Future<String?> getQuiz({
//     required String username,
//     required String email,
//     required String password,
//   }) async {
//     try {
//       final response = await http.post(
//         Uri.parse('$baseUrl/auth/register'),
//         headers: {'Content-Type': 'application/json'},
//         body: jsonEncode({
//           'username': username,
//           'email': email,
//           'password': password,
//         }),
//       );
//
//       if (response.statusCode == 200) {
//         // After successful registration, automatically login
//         return login(username: username, password: password);
//       } else {
//         throw Exception(jsonDecode(response.body)['detail']);
//       }
//     } catch (e) {
//       rethrow;
//     }
//   }
// }