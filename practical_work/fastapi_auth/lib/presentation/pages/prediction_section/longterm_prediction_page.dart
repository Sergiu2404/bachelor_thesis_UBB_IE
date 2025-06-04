import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

class LongTermPredictionPage extends StatefulWidget {
  const LongTermPredictionPage({super.key});

  @override
  State<LongTermPredictionPage> createState() => _LongTermPredictionPageState();
}

class _LongTermPredictionPageState extends State<LongTermPredictionPage> {
  final TextEditingController _tickerController = TextEditingController();
  List<double> predictedPrices = [120, 125, 130, 135, 140, 145, 150, 155, 160, 165];
  List<String> years = [
    '2025', '2026', '2027', '2028', '2029',
    '2030', '2031', '2032', '2033', '2034'
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("10-Year Stock Prediction"),
        backgroundColor: Colors.green.shade700,
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text("Enter Stock Ticker:", style: TextStyle(fontSize: 18)),
            const SizedBox(height: 10),
            TextField(
              controller: _tickerController,
              decoration: InputDecoration(
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                hintText: "e.g. AAPL",
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                // Fetch or mock prediction
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.green.shade800,
                padding: const EdgeInsets.symmetric(vertical: 14),
                minimumSize: const Size(double.infinity, 50),
              ),
              child: const Text("Predict", style: TextStyle(fontSize: 18)),
            ),
            const SizedBox(height: 30),
            const Text("Predicted Prices:", style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            SizedBox(
              height: 250,
              child: LineChart(
                LineChartData(
                  titlesData: FlTitlesData(
                    leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: true)),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          final index = value.toInt();
                          if (index < years.length) {
                            return Text(years[index]);
                          }
                          return const Text("");
                        },
                      ),
                    ),
                  ),
                  gridData: FlGridData(show: true),
                  borderData: FlBorderData(show: true),
                  lineBarsData: [
                    LineChartBarData(
                      spots: List.generate(predictedPrices.length, (i) => FlSpot(i.toDouble(), predictedPrices[i])),
                      isCurved: true,
                      barWidth: 3,
                      color: Colors.green,
                      dotData: FlDotData(show: false),
                    )
                  ],
                ),
              ),
            )
          ],
        ),
      ),
    );
  }
}