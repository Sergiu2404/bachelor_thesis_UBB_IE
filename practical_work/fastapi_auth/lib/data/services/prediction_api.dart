import 'dart:convert';
import 'package:http/http.dart' as http;
import 'dart:developer';

// const String ipv4 = "192.168.1.129"; //acasa
const String ipv4 = "172.30.248.247"; //mateinfo5G

class PredictionService {
  // static const String credibilityBaseUrl = 'http://172.30.248.247:8772'; //mateinfo
  // static const String sentimentBaseUrl = 'http://172.30.248.247:8771';
  // static const String stockPredictionBaseUrl = 'http://172.30.248.247:8773';
  // static const String credibilityBaseUrl = 'http://10.220.19.21:8772'; //fsega
  // static const String sentimentBaseUrl = 'http://10.220.19.21:8771';
  // static const String stockPredictionBaseUrl = 'http://10.220.19.21:8773';
  static const String credibilityBaseUrl = 'http://${ipv4}:8772'; //mateinfo 5G
  static const String sentimentBaseUrl = 'http://${ipv4}:8771';
  static const String stockPredictionBaseUrl = 'http://${ipv4}:8773';



  Future<Map<String, dynamic>> getCredibilityScore(String text) async {
    log("Sending request to credibility API with text: $text");
    final response = await http.post(
      Uri.parse('$credibilityBaseUrl/predict_credibility'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({"text": text}),
    );

    log("Received credibility response: ${response.statusCode} - ${response.body}");

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Credibility prediction failed: ${response.body}");
    }
  }

  Future<Map<String, dynamic>> getSentimentScore(String text) async {
    log("Sending request to sentiment API with text: $text");
    final response = await http.post(
      Uri.parse('$sentimentBaseUrl/predict'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({"text": text}),
    );

    log("Received sentiment response: ${response.statusCode} - ${response.body}");

    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception("Sentiment prediction failed: ${response.body}");
    }
  }

  // Future<List<Map<String, dynamic>>> getAdjustedStockForecast({
  //   required String ticker,
  //   required double sentimentCredibilityScore,
  // }) async {
  //   final uri = Uri.parse('$stockPredictionBaseUrl/predict_stock/').replace(
  //     queryParameters: {
  //       "ticker": ticker,
  //       "adjusted_sentiment_credibility_score": sentimentCredibilityScore.toString(),
  //     },
  //   );
  //
  //   log("Sending request to stock prediction API with URI: $uri");
  //   final response = await http.get(uri);
  //
  //   log("Received stock prediction response: ${response.statusCode} - ${response.body}");
  //
  //   if (response.statusCode == 200) {
  //     final result = json.decode(response.body);
  //     if (result.containsKey("predictions")) {
  //       return List<Map<String, dynamic>>.from(result["predictions"]);
  //     } else {
  //       throw Exception("Invalid stock prediction response");
  //     }
  //   } else {
  //     throw Exception("Stock prediction failed: ${response.body}");
  //   }
  // }
  Future<Map<String, dynamic>> getAdjustedStockForecast({
    required String ticker,
    required double sentimentCredibilityScore,
  }) async {
    final uri = Uri.parse('$stockPredictionBaseUrl/predict_stock/').replace(
      queryParameters: {
        "ticker": ticker,
        "adjusted_sentiment_credibility_score": sentimentCredibilityScore.toString(),
      },
    );

    log("Sending request to stock prediction API with URI: $uri");
    final response = await http.get(uri);

    log("Received stock prediction response: ${response.statusCode} - ${response.body}");

    if (response.statusCode == 200) {
      final result = json.decode(response.body);
      return Map<String, dynamic>.from(result);
    } else {
      throw Exception("Stock prediction failed: ${response.body}");
    }
  }
}
