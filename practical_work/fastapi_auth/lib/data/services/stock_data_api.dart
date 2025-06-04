import 'dart:convert';
import 'dart:developer';
import 'package:fastapi_auth/data/models/stock_data.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:http/http.dart' as http;

class StockDataService {
  //static const String baseUrl = 'http://10.0.2.2:8000';
  //static const String baseUrl = 'http://192.168.1.131:8000';
  static const String baseUrl = 'http://192.168.196.118:8000'; //phone hotspot

  final AuthService _authService = AuthService();

  Future<StockData> getStockForSymbol(String provider, String symbol) async {
    try {
      final token = await _authService.getToken();
      if(token == null){
        throw Exception("User not authenticated");
      }

      final response = await http.get(
        Uri.parse("$baseUrl/stocks/stock/$provider/$symbol"),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token'
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return StockData.fromJson(data);
      } else {
        log("Error fetching stock: ${response.body}");
        throw Exception("Failed to load stock data");
      }
    } catch (e) {
      log("Exception: $e");
      return StockData(companyName: "", symbol: "", price: 0);
    }
  }

  Future<List<StockData>> getStocksForSymbolSubstring(String provider, String symbolSubstring) async {
    try {
      final token = await _authService.getToken();
      if(token == null){
        throw Exception("User not authenticated");
      }

      final response = await http.get(
        Uri.parse("$baseUrl/stocks/$provider?symbol_substr=$symbolSubstring"),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token'
        },
      );

      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);

        return data.map((item) => StockData.fromJson(item)).toList();
      } else {
        log("Error fetching stocks for given substring: ${response.body}");
        throw Exception("Failed to load stocks for given symbol substring");
      }
    } catch (e) {
      log("Exception: $e");
      throw Exception("Failed to load stocks for given symbol substring");
    }
  }


  Future<Map<String, double>> getMonthlyStockData(String provider, String symbol) async {
    try {
      final token = await _authService.getToken();
      if(token == null){
        throw Exception("User not authenticated");
      }

      final response = await http.get(
        Uri.parse("$baseUrl/stocks/monthly/$provider/$symbol"),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $token'
        },
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        if (data.containsKey("monthly_prices")) {
          final Map<String, dynamic> rawPrices = data["monthly_prices"];

          final Map<String, double> formattedPrices = rawPrices.map(
                (key, value) => MapEntry(key, value["Close"]?.toDouble() ?? 0.0),
          );

          return formattedPrices;
        } else {
          throw Exception("Invalid response format");
        }
      } else {
        log("Error fetching monthly stock data: ${response.body}");
        throw Exception("Failed to load monthly stock data");
      }
    } catch (e) {
      log("Exception: $e");
      throw Exception("Failed to load monthly stock data");
    }
  }
}
