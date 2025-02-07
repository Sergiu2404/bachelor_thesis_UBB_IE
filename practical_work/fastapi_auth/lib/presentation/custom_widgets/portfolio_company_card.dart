import 'package:flutter/material.dart';

import '../pages/demo_investing/user_portfolio/buy_sell_page.dart';

class PortfolioCompanyCard extends StatelessWidget {
  final String symbol;
  final String companyName;
  final double currentPrice;
  final int quantity;
  final double currentEvaluation;
  final double averagePurchasePrice;

  const PortfolioCompanyCard({
    super.key,
    required this.symbol,
    required this.companyName,
    required this.currentPrice,
    required this.quantity,
    required this.averagePurchasePrice,
    required this.currentEvaluation,
  });

  @override
  Widget build(BuildContext context) {
      return Card(
        elevation: 4,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                companyName,
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("Current Price:", style: TextStyle(fontSize: 16, color: Colors.grey.shade700)),
                  Text("\$${currentPrice.toStringAsFixed(2)}",
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.green)),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("Quantity:", style: TextStyle(fontSize: 16, color: Colors.grey.shade700)),
                  Text("${quantity}",
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("Average Purchase Price:", style: TextStyle(fontSize: 16, color: Colors.grey.shade700)),
                  Text("\$${averagePurchasePrice.toStringAsFixed(2)}",
                      style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.blue)),
                ],
              ),
              const SizedBox(height: 8),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text("Current Evaluation:", style: TextStyle(fontSize: 16, color: Colors.grey.shade700)),
                  Text(
                    "\$${currentEvaluation.toStringAsFixed(2)}",
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: currentEvaluation >= (quantity * averagePurchasePrice)
                          ? Colors.green
                          : Colors.red,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => BuySellPage(
                            action: 'Buy',
                            symbol: symbol,
                            currentPrice: currentPrice,
                          ),
                        ),
                      );
                    },
                    child: const Text("Buy"),
                  ),
                  ElevatedButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => BuySellPage(
                            action: 'Sell',
                            symbol: symbol,
                            currentPrice: currentPrice,
                          ),
                        ),
                      );
                    },
                    child: const Text("Sell"),
                  ),
                ],
              ),
            ],
          ),
        ),
      );
    }
}
