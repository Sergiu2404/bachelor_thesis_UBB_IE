import 'package:flutter/material.dart';
import '../../../data/services/prediction_api.dart';
import 'adjusted_price_prediction_page.dart';

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

  bool _isChartLoading = false;
  bool _isLoading = false;
  String? _error;

  Future<void> _analyzeArticle() async {
    final article = _newsController.text.trim();

    if (article.length > 500) {
      setState(() => _error = "Article must be 500 characters or less.");
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      if (article.isEmpty) {
        credibilityScore = 1.0;
        sentimentScore = 0.0;
        grammarScore = 1.0;
        punctuationScore = 1.0;
        aiScore = 1.0;
      } else {
        final credibility = await _predictionService.getCredibilityScore(article);
        final sentiment = await _predictionService.getSentimentScore(article);

        credibilityScore = credibility["overall_credibility_score"];
        grammarScore = credibility["detailed_scores"]["grammar_analysis"]["score"];
        punctuationScore = credibility["detailed_scores"]["punctuation_analysis"]["score"];
        aiScore = credibility["detailed_scores"]["ai_model_analysis"]["score"];
        sentimentScore = sentiment["sentiment_score"];
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _isLoading = false);
    }
  }


  // void _goToChartPage() async {
  //   final ticker = _tickerController.text.trim().toUpperCase();
  //   if (ticker.isEmpty) {
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       const SnackBar(content: Text("Please enter a ticker symbol.")),
  //     );
  //     return;
  //   }
  //
  //   final adjustedScore = (credibilityScore * sentimentScore).clamp(-1.0, 1.0);
  //
  //   try {
  //     final response = await _predictionService.getAdjustedStockForecast(
  //       ticker: ticker,
  //       sentimentCredibilityScore: adjustedScore,
  //     );
  //
  //     final double? currentPrice = (response["current_price"] as num?)?.toDouble();
  //     final predictions = response["predictions"];
  //
  //     if (currentPrice == null || predictions == null || predictions.isEmpty) {
  //       ScaffoldMessenger.of(context).showSnackBar(
  //         SnackBar(content: Text("Ticker '$ticker' not found or data unavailable.")),
  //       );
  //       return;
  //     }
  //
  //     Navigator.push(
  //       context,
  //       MaterialPageRoute(
  //         builder: (_) => AdjustedPricePredictionPage(
  //           ticker: ticker,
  //           sentimentCredibilityScore: adjustedScore,
  //         ),
  //       ),
  //     );
  //   } catch (e) {
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       SnackBar(content: Text("Error: ${e.toString()}")),
  //     );
  //   }
  // }

  void _goToChartPage() async {
    final ticker = _tickerController.text.trim().toUpperCase();
    if (ticker.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Please enter a ticker symbol.")),
      );
      return;
    }

    final adjustedScore = (credibilityScore * sentimentScore).clamp(-1.0, 1.0);

    setState(() => _isChartLoading = true); // start loading

    try {
      final response = await _predictionService.getAdjustedStockForecast(
        ticker: ticker,
        sentimentCredibilityScore: adjustedScore,
      );

      final double? currentPrice = (response["current_price"] as num?)?.toDouble();
      final predictions = response["predictions"];

      if (currentPrice == null || predictions == null || predictions.isEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Ticker '$ticker' not found or data unavailable.")),
        );
        return;
      }

      setState(() => _isChartLoading = false);

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (_) => AdjustedPricePredictionPage(
            ticker: ticker,
            sentimentCredibilityScore: adjustedScore,
          ),
        ),
      );
    } catch (e) {
      setState(() => _isChartLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: ${e.toString()}")),
      );
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Analyze News Sentiment"),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pushNamedAndRemoveUntil(context, '/home', (route) => false);
          },
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Enter Related News Article (max 500 chars):", style: TextStyle(fontSize: 18)),
            const SizedBox(height: 10),
            TextField(
              controller: _newsController,
              maxLines: 5,
              maxLength: 500,
              decoration: InputDecoration(
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                hintText: "Paste article content here...",
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _isLoading ? null : _analyzeArticle,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.indigo.shade700,
                minimumSize: const Size(double.infinity, 50),
              ),
              child: _isLoading
                  ? const CircularProgressIndicator(color: Colors.white)
                  : const Text("Analyze Article", style: TextStyle(fontSize: 18, color: Colors.white)),
            ),
            const SizedBox(height: 30),
            if (_error != null)
              Text(_error!, style: const TextStyle(color: Colors.red)),

            //if (credibilityScore > 0 || sentimentScore != 0.0)
            if (!_isLoading) ...[
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
                      Text(" - Grammar Score: ${(grammarScore * 100).toStringAsFixed(1)}%", style: const TextStyle(fontSize: 16)),
                      Text(" - Punctuation Score: ${(punctuationScore * 100).toStringAsFixed(1)}%", style: const TextStyle(fontSize: 16)),
                      Text(" - AI Model Score: ${(aiScore * 100).toStringAsFixed(1)}%", style: const TextStyle(fontSize: 16)),
                      const SizedBox(height: 10),
                      Text("Sentiment Score: ${sentimentScore.toStringAsFixed(2)}",
                          style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 30),
              const Text("Enter Ticker for Prediction:", style: TextStyle(fontSize: 18)),
              const SizedBox(height: 10),
              TextField(
                controller: _tickerController,
                decoration: InputDecoration(
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                  hintText: "ex: AAPL",
                ),
              ),
              const SizedBox(height: 20),
              // ElevatedButton(
              //   onPressed: _goToChartPage,
              //   style: ElevatedButton.styleFrom(
              //     backgroundColor: Colors.green.shade600,
              //     minimumSize: const Size(double.infinity, 50),
              //   ),
              //   child: const Text("View Adjusted Price Chart", style: TextStyle(fontSize: 18, color: Colors.white)),
              // ),
              ElevatedButton(
                onPressed: _isChartLoading ? null : _goToChartPage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green.shade600,
                  minimumSize: const Size(double.infinity, 50),
                ),
                child: _isChartLoading
                    ? const SizedBox(
                  height: 24,
                  width: 24,
                  child: CircularProgressIndicator(
                    color: Colors.white,
                    strokeWidth: 2,
                  ),
                )
                    : const Text("View Adjusted Price Chart", style: TextStyle(fontSize: 18, color: Colors.white)),
              ),
              const SizedBox(height: 30),
            ]
          ],
        ),
      ),
    );
  }
}
