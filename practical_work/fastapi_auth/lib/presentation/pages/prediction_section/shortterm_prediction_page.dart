import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../../../data/services/prediction_api.dart';

class ShortTermPredictionPage extends StatefulWidget {
  const ShortTermPredictionPage({super.key});

  @override
  State<ShortTermPredictionPage> createState() => _ShortTermPredictionPageState();
}

class _ShortTermPredictionPageState extends State<ShortTermPredictionPage> {
  final TextEditingController _newsController = TextEditingController();
  final TextEditingController _tickerController = TextEditingController();
  final PredictionService _predictionService = PredictionService();

  double credibilityScore = 0.0;
  double sentimentScore = 0.0;
  double grammarScore = 0.0;
  double punctuationScore = 0.0;
  double aiScore = 0.0;
  List<double> adjustedPrices = [];
  List<String> days = [];

  bool _isLoading = false;
  String? _error;

  Future<void> _fetchPredictions() async {
    final article = _newsController.text.trim();
    final ticker = _tickerController.text.trim().toUpperCase();

    setState(() {
      _isLoading = true;
      _error = null;
      adjustedPrices.clear();
      days.clear();
    });

    try {
      if (article.isNotEmpty) {
        final credibility = await _predictionService.getCredibilityScore(article);
        final sentiment = await _predictionService.getSentimentScore(article);

        credibilityScore = credibility["overall_credibility_score"];
        grammarScore = credibility["detailed_scores"]["grammar_analysis"]["score"];
        punctuationScore = credibility["detailed_scores"]["punctuation_analysis"]["score"];
        aiScore = credibility["detailed_scores"]["ai_model_analysis"]["score"];
        sentimentScore = sentiment["sentiment_score"];
      }

      if (ticker.isNotEmpty) {
        final adjusted = (credibilityScore * sentimentScore).clamp(-1.0, 1.0);
        final predictions = await _predictionService.getAdjustedStockForecast(
          ticker: ticker,
          sentimentCredibilityScore: article.isNotEmpty ? adjusted : 0.0,
        );

        adjustedPrices = predictions.map((p) => (p["predicted_close"] as num).toDouble()).toList();
        days = predictions.map((p) => DateFormat("MMM d").format(DateTime.parse(p["date"]))).toList();
      }

      setState(() {});
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("News-Based 10-Day Prediction"),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Enter Related News Article:", style: TextStyle(fontSize: 18)),
            const SizedBox(height: 10),
            TextField(
              controller: _newsController,
              maxLines: 5,
              maxLength: 500,
              decoration: InputDecoration(
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                hintText: "Paste article content here (max 500 characters)...",
              ),
            ),
            const SizedBox(height: 20),
            const Text("Enter Company Ticker:", style: TextStyle(fontSize: 18)),
            const SizedBox(height: 10),
            TextField(
              controller: _tickerController,
              decoration: InputDecoration(
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                hintText: "e.g. TSLA",
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isLoading ? null : _fetchPredictions,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.indigo.shade700,
                minimumSize: const Size(double.infinity, 50),
              ),
              child: _isLoading
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text("Analyze and Predict", style: TextStyle(fontSize: 18)),
            ),
            const SizedBox(height: 30),
            if (_error != null)
              Text(_error!, style: const TextStyle(color: Colors.red)),

            if (credibilityScore > 0 || sentimentScore != 0.0) ...[
              const Text("\nAnalysis Results:", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              Card(
                color: Colors.indigo.shade50,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text("Credibility Score: ${(credibilityScore * 100).toStringAsFixed(1)}%",
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                      Text(" - Grammar Score: ${(grammarScore * 100).toStringAsFixed(1)}%",
                          style: const TextStyle(fontSize: 16)),
                      Text(" - Punctuation Score: ${(punctuationScore * 100).toStringAsFixed(1)}%",
                          style: const TextStyle(fontSize: 16)),
                      Text(" - AI Model Score: ${(aiScore * 100).toStringAsFixed(1)}%",
                          style: const TextStyle(fontSize: 16)),
                      const SizedBox(height: 10),
                      Text("Sentiment Score: ${sentimentScore.toStringAsFixed(2)}",
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              )
            ],

            if (adjustedPrices.isNotEmpty) ...[
              const SizedBox(height: 20),
              const Text("Adjusted Price Forecast:", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              SizedBox(
                height: 250,
                child: LineChart(
                  LineChartData(
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
                            if (index < days.length) {
                              return Padding(
                                padding: const EdgeInsets.only(top: 8.0),
                                child: Text(days[index], style: const TextStyle(fontSize: 10)),
                              );
                            }
                            return const Text("");
                          },
                        ),
                      ),
                      leftTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 40,
                          interval: 10,
                          getTitlesWidget: (value, meta) => Text(value.toStringAsFixed(0), style: const TextStyle(fontSize: 10)),
                        ),
                      ),
                    ),
                    gridData: FlGridData(show: true),
                    borderData: FlBorderData(show: true),
                    lineBarsData: [
                      LineChartBarData(
                        spots: List.generate(adjustedPrices.length, (i) => FlSpot(i.toDouble(), adjustedPrices[i])),
                        isCurved: true,
                        barWidth: 3,
                        color: Colors.indigo,
                        dotData: FlDotData(show: false),
                      )
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 100),
            ]
          ],
        ),
      ),
    );
  }
}
