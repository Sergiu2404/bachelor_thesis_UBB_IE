import 'dart:developer';

import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:fastapi_auth/data/models/stock_data.dart';
import 'package:fastapi_auth/data/services/stock_data_api.dart';

import '../user_portfolio/buy_sell_page.dart';

class StockDetailsPage extends StatefulWidget {
  final StockData stockData;

  const StockDetailsPage({super.key, required this.stockData});

  @override
  State<StockDetailsPage> createState() => _StockDetailsPageState();
}

class _StockDetailsPageState extends State<StockDetailsPage> {
  final StockDataService _stockService = StockDataService();

  StockData? stockDetails;
  Map<String, double> monthlyPrices = {};
  bool isLoading = true;

  static const int minDataPointsForChart = 3;

  @override
  void initState() {
    super.initState();
    _fetchStockDetails();
  }

  Future<void> _fetchStockDetails() async {
    try {
      final stock = await _stockService.getStockForSymbol("yahoo", widget.stockData.symbol);
      final monthly = await _stockService.getMonthlyStockData("yahoo", widget.stockData.symbol);
      log("FETCHED PRICE AND SYMBOL: ${stock.price} for ${stock.symbol}");
      log("Monthly data points: ${monthly.length}");

      setState(() {
        stockDetails = stock;
        monthlyPrices = monthly;
        isLoading = false;
      });
    } catch (exception) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: ${exception.toString()}")),
      );
      setState(() => isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.stockData.symbol),
        backgroundColor: Colors.blueAccent,
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Stock: ${stockDetails?.companyName ?? "N/A"}",
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            Text("Symbol: ${stockDetails?.symbol ?? "N/A"}",
                style: const TextStyle(fontSize: 16)),
            const SizedBox(height: 10),
            Text("Price: \$${stockDetails?.price.toStringAsFixed(2) ?? "N/A"}",
                style: const TextStyle(fontSize: 16)),
            const SizedBox(height: 20),

            monthlyPrices.length >= minDataPointsForChart
                ? SizedBox(
              height: 300,
              child: _buildStockChart(),
            )
                : Padding(
              padding: const EdgeInsets.symmetric(vertical: 20.0),
              child: Text(
                "Not enough historical data to display a chart.",
                style: TextStyle(
                  color: Colors.grey[600],
                  fontStyle: FontStyle.italic,
                  fontSize: 16,
                ),
              ),
            ),

            const SizedBox(height: 20),

            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => BuySellPage(
                          action: "Buy",
                          symbol: widget.stockData.symbol,
                          currentPrice: stockDetails?.price ?? 0.0,
                        ),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
                  child: const Text("Buy", style: TextStyle(fontSize: 18)),
                ),
                const SizedBox(width: 20),
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => BuySellPage(
                          action: "Sell",
                          symbol: widget.stockData.symbol,
                          currentPrice: stockDetails?.price ?? 0.0,
                        ),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                  child: const Text("Sell", style: TextStyle(fontSize: 18)),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStockChart() {
    final List<FlSpot> spots = [];
    final sortedKeys = monthlyPrices.keys.toList()..sort();

    for (int i = 0; i < sortedKeys.length; i++) {
      final month = sortedKeys[i];
      final closePrice = monthlyPrices[month] ?? 0.0;
      spots.add(FlSpot(i.toDouble(), closePrice));
    }

    final prices = spots.map((e) => e.y).toList();
    final minY = (prices.reduce((a, b) => a < b ? a : b) - 10).floorToDouble();
    final maxY = (prices.reduce((a, b) => a > b ? a : b) + 10).ceilToDouble();
    final range = maxY - minY;
    final interval = (range <= 20 ? 5 : range <= 50 ? 10 : 20).toDouble();

    return LineChart(
      LineChartData(
        minY: minY,
        maxY: maxY,
        gridData: FlGridData(show: true),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 50,
              interval: interval,
              getTitlesWidget: (value, meta) => Text(
                value.toStringAsFixed(0),
                style: const TextStyle(fontSize: 10),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              getTitlesWidget: (value, meta) {
                if (value.toInt() >= 0 && value.toInt() < sortedKeys.length) {
                  final dateStr = sortedKeys[value.toInt()]; // e.g., "2024-07"
                  final parts = dateStr.split('-');
                  final year = parts[0];
                  final monthNum = int.parse(parts[1]);
                  const monthNames = [
                    '', // 0 index unused
                    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
                  ];
                  final monthName = monthNames[monthNum];
                  return Text(
                    '$monthName $year',
                    style: const TextStyle(fontSize: 7),
                  );
                }
                return const Text("");
              },
            ),
          ),
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        borderData: FlBorderData(show: true),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            color: Colors.blue,
            barWidth: 3,
            belowBarData: BarAreaData(
              show: true,
              color: Colors.blue.withOpacity(0.3),
            ),
            dotData: FlDotData(show: false),
          ),
        ],
      ),
    );
    }
  }
