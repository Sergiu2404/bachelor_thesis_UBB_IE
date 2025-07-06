import 'dart:developer';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import '../../../data/services/prediction_api.dart';

class AdjustedPricePredictionPage extends StatefulWidget {
  final String ticker;
  final double sentimentCredibilityScore;

  const AdjustedPricePredictionPage({
    super.key,
    required this.ticker,
    required this.sentimentCredibilityScore,
  });

  @override
  State<AdjustedPricePredictionPage> createState() => _AdjustedPricePredictionPageState();
}

class _AdjustedPricePredictionPageState extends State<AdjustedPricePredictionPage> {
  final PredictionService _predictionService = PredictionService();

  List<double> adjustedPrices = [];
  List<String> days = [];
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    //Future.delayed(Duration.zero, () => _fetchChartData());
    _fetchChartData();
  }

  // Future<void> _fetchChartData() async {
  //   try {
  //     final response = await _predictionService.getAdjustedStockForecast(
  //       ticker: widget.ticker,
  //       sentimentCredibilityScore: widget.sentimentCredibilityScore,
  //     );
  //
  //     final double? currentPrice = (response["current_price"] as num?)?.toDouble();
  //     log("CURRENT PRICE FETCHED: ${currentPrice}");
  //     final List predictions = List<Map<String, dynamic>>.from(response["predictions"]);
  //
  //     List<double> prices = predictions.map((p) => (p["predicted_close"] as num).toDouble()).toList();
  //
  //     if (currentPrice != null) {
  //       prices.insert(0, currentPrice);
  //     } else {
  //       debugPrint("currentPrice is null, skipping prepend.");
  //     }
  //
  //     setState(() {
  //       adjustedPrices = prices;
  //       days = _generateBusinessDays(count: prices.length);
  //       _isLoading = false;
  //     });
  //   } catch (e) {
  //     log("Error fetching chart data: $e");
  //     setState(() {
  //       _error = e.toString();
  //       _isLoading = false;
  //     });
  //   }
  // }
  Future<void> _fetchChartData() async {
    try {
      final response = await _predictionService.getAdjustedStockForecast(
        ticker: widget.ticker,
        sentimentCredibilityScore: widget.sentimentCredibilityScore,
      );

      if (response["current_price"] == null || response["predictions"] == null) {
        throw Exception("Ticker '${widget.ticker}' not found or data unavailable.");
      }

      final double? currentPrice = (response["current_price"] as num?)?.toDouble();
      final List predictions = List<Map<String, dynamic>>.from(response["predictions"]);

      List<double> prices = predictions.map((p) {
        final num? value = p["predicted_close"] as num?;
        if (value == null) throw Exception("Prediction data incomplete.");
        return value.toDouble();
      }).toList();

      if (currentPrice != null) {
        prices.insert(0, currentPrice);
      }

      setState(() {
        adjustedPrices = prices;
        days = _generateBusinessDays(count: prices.length);
        _isLoading = false;
      });
    } catch (e) {
      log("Error fetching chart data: $e");
      setState(() {
        _error = "Error: ${e.toString().replaceAll('Exception: ', '')}";
        _isLoading = false;
      });
    }
  }


  List<String> _generateBusinessDays({int count = 11}) {
    List<String> businessDays = [];
    DateTime current = DateTime.now();
    int added = 0;

    while (added < count) {
      if (current.weekday != DateTime.saturday && current.weekday != DateTime.sunday) {
        businessDays.add(DateFormat("MMM d").format(current));
        added++;
      }
      current = current.add(const Duration(days: 1));
    }

    return businessDays;
  }

  @override
  Widget build(BuildContext context) {
    const double yMargin = 5.0;

    Widget body;

    if (_isLoading) {
      body = const Center(child: CircularProgressIndicator());
    } else if (_error != null) {
      body = Center(child: Text(_error!, style: const TextStyle(color: Colors.red)));
    } else if (adjustedPrices.isEmpty) {
      body = const Center(child: Text("No price data available."));
    } else {
      final double rawMin = adjustedPrices.reduce((a, b) => a < b ? a : b);
      final double rawMax = adjustedPrices.reduce((a, b) => a > b ? a : b);
      final double minY = (rawMin - yMargin).floorToDouble();
      final double maxY = (rawMax + yMargin).ceilToDouble();
      final double minX = 0;
      final double maxX = adjustedPrices.length.toDouble() - 1;
      final double yRange = maxY - minY;
      final double interval = yRange <= 20 ? 5 : yRange <= 50 ? 10 : 20;

      body = Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Stock Price Forecast:", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 50),
            Expanded(
              child: LineChart(
                LineChartData(
                  minY: minY,
                  maxY: maxY,
                  minX: minX,
                  maxX: maxX,
                  lineTouchData: LineTouchData(
                    handleBuiltInTouches: true,
                    touchTooltipData: LineTouchTooltipData(
                      getTooltipItems: (touchedSpots) {
                        return touchedSpots.map((spot) {
                          final index = spot.x.toInt();
                          return LineTooltipItem(
                            '${days[index]}\n\$${spot.y.toStringAsFixed(2)}',
                            const TextStyle(
                              color: Colors.black,
                              fontWeight: FontWeight.bold,
                              backgroundColor: Colors.white,
                            ),
                          );
                        }).toList();
                      },
                    ),
                  ),
                  titlesData: FlTitlesData(
                    topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 32,
                        interval: 1,
                        getTitlesWidget: (value, meta) {
                          final index = value.toInt();
                          if (index < days.length && index % 2 == 0) {
                            return Padding(
                              padding: const EdgeInsets.only(top: 8.0),
                              child: Text(
                                index == 0 ? "Today" : days[index],
                                style: TextStyle(
                                  fontSize: 10,
                                  fontWeight: index == 0 ? FontWeight.bold : FontWeight.normal,
                                  color: index == 0 ? Colors.red : Colors.black,
                                ),
                              ),
                            );
                          }
                          return const SizedBox.shrink();
                        },
                      ),
                    ),
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        interval: interval,
                        getTitlesWidget: (value, meta) => Text(
                          value.toStringAsFixed(0),
                          style: const TextStyle(fontSize: 10),
                        ),
                      ),
                    ),
                  ),
                  gridData: FlGridData(show: true),
                  borderData: FlBorderData(show: true),
                  lineBarsData: [
                    LineChartBarData(
                      spots: List.generate(
                        adjustedPrices.length,
                            (i) => FlSpot(i.toDouble(), adjustedPrices[i]),
                      ),
                      isCurved: true,
                      barWidth: 3,
                      color: Colors.indigo,
                      dotData: FlDotData(show: true),
                    )
                  ],
                ),
              ),
            ),
            const SizedBox(height: 50),
          ],
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text("Adjusted Price Forecast"),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
          },
        ),
      ),
      backgroundColor: Colors.white,
      body: body,
    );
  }
}
