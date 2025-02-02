import 'dart:convert';
import 'dart:developer' as developer;
import 'package:http/http.dart' as http;

class StocksServiceAPI {
  final String apiKey = 'I0OY1MTU01Z74V0E';
  final List<String> stockSymbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];

  Future<List<Map<String, dynamic>>> fetchStocksBySymbol(String searchTerm) async {
    final String baseUrl = 'https://www.alphavantage.co/query';
    List<Map<String, dynamic>> stockData = [];

    // First, search for matching symbols
    final searchUri = Uri.parse(
      '$baseUrl?function=SYMBOL_SEARCH&keywords=$searchTerm&apikey=$apiKey',
    );

    try {
      final searchResponse = await http.get(searchUri);
      developer.log('Search response status: ${searchResponse.statusCode}');

      if (searchResponse.statusCode == 200) {
        final searchData = json.decode(searchResponse.body);

        if (searchData.containsKey('bestMatches')) {
          final matches = searchData['bestMatches'] as List;
          developer.log('Found ${matches.length} matching symbols');

          // For each matching symbol, fetch its current price data
          for (var match in matches) {
            final symbol = match['1. symbol'] as String;
            final name = match['2. name'] as String;
            final region = match['4. region'] as String;

            // Only process US stocks to avoid formatting issues
            if (region == 'United States') {
              final priceData = await _fetchStockPrice(symbol);

              if (priceData != null) {
                stockData.add({
                  'name': name,
                  'symbol': symbol,
                  'region': region,
                  'lastUpdated': priceData['lastUpdated'],
                  'currentPrice': priceData['currentPrice'],
                });
              }
            }
          }
        } else if (searchData.toString().contains('Thank you for using Alpha Vantage!')) {
          throw Exception('API rate limit reached. Please try again in a minute.');
        }
      } else {
        throw Exception('Failed to search for stocks. Status: ${searchResponse.statusCode}');
      }
    } catch (e) {
      developer.log('Error in fetchStocksBySymbol: $e');
      rethrow;
    }

    return stockData;
  }

  Future<Map<String, dynamic>?> _fetchStockPrice(String symbol) async {
    final String baseUrl = 'https://www.alphavantage.co/query';
    final uri = Uri.parse(
      '$baseUrl?function=TIME_SERIES_INTRADAY&symbol=$symbol&interval=5min&apikey=$apiKey&outputsize=compact&datatype=json',
    );

    try {
      final response = await http.get(uri);
      developer.log('Price fetch response status for $symbol: ${response.statusCode}');

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        if (data.containsKey('Meta Data') && data.containsKey('Time Series (5min)')) {
          final timeSeries = data['Time Series (5min)'] as Map<String, dynamic>;
          final latestEntry = timeSeries.entries.first;

          return {
            'lastUpdated': latestEntry.key,
            'currentPrice': double.parse(latestEntry.value['1. open']),
          };
        }
      }
    } catch (e) {
      developer.log('Error fetching price for $symbol: $e');
    }

    return null;
  }

  // Fetch data for predefined stock list
  Future<List<Map<String, dynamic>>> fetchStockData() async {
    List<Map<String, dynamic>> stockData = [];

    for (String symbol in stockSymbols) {
      final priceData = await _fetchStockPrice(symbol);

      if (priceData != null) {
        stockData.add({
          'name': symbol,
          'lastUpdated': priceData['lastUpdated'],
          'currentPrice': priceData['currentPrice'],
        });
      }
    }

    return stockData;
  }
}