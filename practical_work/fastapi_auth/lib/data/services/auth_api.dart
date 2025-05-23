// lib/services/auth_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';


class AuthService {
  //static const String baseUrl = 'http://10.0.2.2:8000'; //emulator
  //static const String baseUrl = 'http://192.168.1.131:8000'; // phone retea acasa
  static const String baseUrl = 'http://192.168.51.118:8000'; // phone hotspot
  static const String tokenKey = 'auth_token';

  Future<String?> register({
    required String username,
    required String email,
    required String password,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/auth/register'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'username': username,
          'email': email,
          'password': password,
        }),
      );

      if (response.statusCode == 200) {
        // After successful registration, automatically login
        return login(username: username, password: password);
      } else {
        throw Exception(jsonDecode(response.body)['detail']);
      }
    } catch (e) {
      rethrow;
    }
  }

  Future<String?> login({
    required String username,
    required String password,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'username': username,
          'password': password,
        }),
      );

      if (response.statusCode == 200) {
        final token = jsonDecode(response.body)['access_token'];
        await _saveToken(token);
        return token;
      } else {
        throw Exception(jsonDecode(response.body)['detail']);
      }
    } catch (e) {
      rethrow;
    }
  }

  Future<void> updateUserBalance(double newBalance) async {
    final token = await getToken();
    if(token == null)
      return;

    try{
      final response = await http.post(
        Uri.parse("$baseUrl/auth/update-balance"),
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer $token"
        },
        body: json.encode({"new_balance": newBalance})
      );

      if(response.statusCode != 200){
        throw Exception("Failed to update balance");
      }
    } catch(exception) {
      print("Error updating the balance: $exception");
    }
  }
  
  Future<Map<String, dynamic>> getCurrentUser() async {
    final token = await getToken();
    if(token == null)
      return {
        "username": "guest",
        "email": "",
        "virtual_money_balance": 0
      };
    
    try{
      final response = await http.get(
        Uri.parse('$baseUrl/auth/connected-user'),
        headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $token',
        },
      );

      if(response.statusCode == 200){
        return jsonDecode(response.body);
      } else {
        return {
          "username": "guest",
          "email": "",
          "virtual_money_balance": 0
        };
      }
    } catch(error){
      return {
        "username": "guest",
        "email": "",
        "virtual_money_balance": 0
      };
    }
  }

  // Future<void> logout() async {
  //   final prefs = await SharedPreferences.getInstance();
  //   await prefs.remove(tokenKey);
  // }

  Future<void> logout() async {
    final prefs = await SharedPreferences.getInstance();
    final token = await getToken();

    if (token != null) {
      try {
        final response = await http.post(
          Uri.parse('$baseUrl/auth/logout'), // Make sure this endpoint exists in your FastAPI backend
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer $token',
          },
        );

        if (response.statusCode == 200) {
          await prefs.remove(tokenKey);
        } else {
          throw Exception("Logout failed on server");
        }
      } catch (e) {
        print("Error logging out: $e");
      }
    }

    await prefs.remove(tokenKey);
  }



  Future<void> _saveToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(tokenKey, token);
  }

  Future<String?> getToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(tokenKey);
  }
}