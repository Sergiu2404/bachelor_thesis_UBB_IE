import 'package:fastapi_auth/presentation/pages/prediction_section/longterm_prediction_page.dart';
import 'package:fastapi_auth/presentation/pages/prediction_section/shortterm_prediction_page.dart';
import 'package:flutter/material.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';


class PredictionMainPage extends StatefulWidget {
  const PredictionMainPage({super.key});

  @override
  State<PredictionMainPage> createState() => _PredictionMainPageState();
}

class _PredictionMainPageState extends State<PredictionMainPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("AI Stock Predictor",
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
        backgroundColor: Colors.deepPurple,
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const Text(
              "Select Prediction Type",
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 30),

            ElevatedButton(
              onPressed: () {
                showPredictionConfirmationDialog(
                  context: context,
                  onConfirm: () {
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(builder: (context) => ShortTermPredictionPage()),
                    );
                  },
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.indigo,
                minimumSize: const Size(double.infinity, 60),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text(
                "📄 News-Based Prediction (10 Days)",
                style: TextStyle(fontSize: 18, color: Colors.white),
              ),
            ),
            const SizedBox(height: 20),

            ElevatedButton(
              onPressed: () {
                showPredictionConfirmationDialog(
                  context: context,
                  onConfirm: () {
                    Navigator.pushReplacement(
                      context,
                      MaterialPageRoute(builder: (context) => LongTermPredictionPage()),
                    );
                  },
                );
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green,
                minimumSize: const Size(double.infinity, 60),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text(
                "📈 Long-Term Stock Prediction (10 Years)",
                style: TextStyle(fontSize: 18, color: Colors.white),
              ),
            ),

            const SizedBox(height: 40),
            Container(
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.blueGrey.shade50,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: const Text(
                "The first prediction uses a short news article (max 500 chars) and a stock ticker. It returns a credibility score (0-100%) and a sentiment score (-1 to 1), then forecasts the stock's adjusted price over the next 10 days.\n\nThe second predicts the long-term price over 10 years based only on historical performance, ignoring news sentiment.",
                style: TextStyle(fontSize: 16, height: 1.5),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

Future<void> showPredictionConfirmationDialog({
  required BuildContext context,
  required Function onConfirm,
}) async {
  final authService = AuthService();
  final user = await authService.getCurrentUser();
  final double balance = user["virtual_money_balance"] ?? 0.0;

  if (balance < 1.0) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Insufficient Balance"),
        content: const Text("You need at least \$1.00 virtual money to request this prediction."),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("OK"),
          ),
        ],
      ),
    );
    return;
  }

  showDialog(
    context: context,
    builder: (BuildContext ctx) {
      return AlertDialog(
        title: const Text("Prediction Cost"),
        content: const Text("This prediction costs \$1.00 of your virtual money. Proceed?"),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text("Cancel", style: TextStyle(color: Colors.red)),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              await authService.updateUserBalance(balance - 1.0);
              onConfirm();
            },
            child: const Text("Yes, Proceed"),
          ),
        ],
      );
    },
  );
}
