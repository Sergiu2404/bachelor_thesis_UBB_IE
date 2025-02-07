import 'dart:convert';
import 'dart:developer';
import 'package:fastapi_auth/data/models/portfolio_company.dart';
import 'package:http/http.dart'as http;
import 'package:fastapi_auth/data/services/auth_api.dart';

class PortfolioService{
  static const String baseUrl = 'http://10.0.2.2:8000/portfolio';
  final AuthService _authService = AuthService();

  Future<List<PortfolioCompany>> getPortfolioForUser() async {
    Map<String, dynamic>? currentUser = await _authService.getCurrentUser();
    log("Fetching portfolio user: ${currentUser?["username"]}");

    try {
      final response = await http.get(
          Uri.parse("$baseUrl/${currentUser?["username"]}"),
          headers: {
            'Content-Type': 'application/json'
          });

      if(response.statusCode == 200){
        List<dynamic> jsonResponse = json.decode(response.body);

        return jsonResponse.map((item) => PortfolioCompany(
          username: item["username"],
          symbol: item["symbol"],
          companyName: item["company_name"],
          quantity: (item["quantity"] ?? 0).toInt(),
          averageBuyPrice: (item["average_buy_price"] ?? 0).toDouble(),
          totalCurrentValue: (item["total_current_price"] ?? 0).toDouble()
        )).toList();
      } else {
        log("Error: ${response.statusCode}, ${response.body}");
        return [];
      }
    } catch(exception){
      log("Exception: $exception");
      return [];
    }
  }

  Future<Map<String, dynamic>> buyStockForUser(String symbol, int quantity) async {
    Map<String, dynamic>? currentUser = await _authService.getCurrentUser();
    String username = currentUser?["username"] ?? "";

    try {
      final response = await http.post(
        Uri.parse("$baseUrl/$username/buy"),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          "symbol": symbol,
          "quantity": quantity,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        log("Error buying stock: ${response.statusCode}, ${response.body}");
        return {
          "error": "Failed to buy stock",
          "status": response.statusCode,
          "message": response.body,
        };
      }
    } catch (exception) {
      log("Exception while buying stock: $exception");
      return {"error": "Exception occurred while buying stock"};
    }
  }

  /// **Sell Stock for User**
  Future<Map<String, dynamic>> sellStockForUser(String symbol, int quantity) async {
    Map<String, dynamic>? currentUser = await _authService.getCurrentUser();
    String username = currentUser?["username"] ?? "";

    try {
      final response = await http.post(
        Uri.parse("$baseUrl/$username/sell"),
        headers: {
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          "symbol": symbol,
          "quantity": quantity,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        log("Error selling stock: ${response.statusCode}, ${response.body}");
        return {
          "error": "Failed to sell stock",
          "status": response.statusCode,
          "message": response.body,
        };
      }
    } catch (exception) {
      log("Exception while selling stock: $exception");
      return {"error": "Exception occurred while selling stock"};
    }
  }
}