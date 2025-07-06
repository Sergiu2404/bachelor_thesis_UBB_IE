import 'dart:convert';
import 'dart:developer';
import 'package:fastapi_auth/data/models/stock_data.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:http/http.dart' as http;

// const String ipv4 = "192.168.1.129"; //acasa
//const String ipv4 = "10.220.19.21"; //fsega
const String ipv4 = "172.30.248.247"; //mateinfo5G


class StockDataService {
  //static const String baseUrl = 'http://10.0.2.2:8000';
  // static const String baseUrl = 'http://192.168.233.118:8244'; //phone hotspot
  static const String baseUrl = 'http://${ipv4}:8244';

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

        log("API Response: $data");

        if (data.containsKey("error")) {
          throw Exception("API Error: ${data["error"]}");
        }

        final stockData = StockData(
          companyName: data["company_name"] ?? data["companyName"] ?? "N/A",
          symbol: data["symbol"] ?? "N/A",
          price: (data["latest_price"] ?? data["price"] ?? 0).toDouble(),
        );

        log("Mapped StockData: ${stockData.symbol}, ${stockData.price}");
        return stockData;

      } else {
        log("HTTP Error ${response.statusCode}: ${response.body}");
        throw Exception("Failed to load stock data: HTTP ${response.statusCode}");
      }
    } catch (e) {
      log("Exception in getStockForSymbol: $e");
      rethrow;
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
