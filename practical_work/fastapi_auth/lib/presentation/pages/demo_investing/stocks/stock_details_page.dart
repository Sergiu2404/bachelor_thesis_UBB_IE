import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:fastapi_auth/data/models/stock_data.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:fastapi_auth/data/services/portfolio_api.dart';
import 'package:fastapi_auth/data/services/stock_data_api.dart';

class StockDetailsPage extends StatefulWidget {
  final StockData stockData;

  const StockDetailsPage({super.key, required this.stockData});

  @override
  State<StockDetailsPage> createState() => _StockDetailsPageState();
}

class _StockDetailsPageState extends State<StockDetailsPage> {
  final StockDataService _stockService = StockDataService();
  final AuthService _authService = AuthService();
  final PortfolioService _portfolioService = PortfolioService();

  Map<String, dynamic> currentConnectedUser = {};

  String providerAPI = "yahoo";
  final List<String> providerOptions = ["yahoo", "alpha"];

  StockData? stockDetails;
  Map<String, Map<String, double>>? monthlyPrices;
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchStockDetails();
  }

  Future<void> _fetchStockDetails() async {
    try {
      final stock = await _stockService.getStockForSymbol(providerAPI, widget.stockData.symbol);
      final monthly = await _stockService.getMonthlyStockData(providerAPI, widget.stockData.symbol);

      setState(() {
        stockDetails = stock;
        monthlyPrices = monthly;
        isLoading = false;
      });
    } catch (exception) {
      ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Error: ${exception.toString()}"))
      );
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.stockData.symbol)),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Company: ${stockDetails?.companyName ?? "N/A"}",
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 10),
            Text("Symbol: ${stockDetails?.symbol ?? "N/A"}",
                style: const TextStyle(fontSize: 16)),
            const SizedBox(height: 10),
            Text("Price: \$${stockDetails?.price.toStringAsFixed(2) ?? "N/A"}",
                style: const TextStyle(fontSize: 16)),
            const SizedBox(height: 20),

            if (monthlyPrices != null && monthlyPrices!.isNotEmpty)
              SizedBox(
                height: 300,
                child: _buildStockChart(),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStockChart() {
    final List<FlSpot> spots = [];
    final sortedKeys = monthlyPrices!.keys.toList()..sort();

    for (int i = 0; i < sortedKeys.length; i++) {
      final month = sortedKeys[i];
      final closePrice = monthlyPrices![month]?["Close"] ?? 0;
      spots.add(FlSpot(i.toDouble(), closePrice));
    }

    return LineChart(
      LineChartData(
        gridData: FlGridData(show: false),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(showTitles: true, reservedSize: 40),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (value, meta) {
                if (value.toInt() >= 0 && value.toInt() < sortedKeys.length) {
                  return Text(sortedKeys[value.toInt()].substring(5));
                }
                return const Text("");
              },
              reservedSize: 30,
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
            belowBarData: BarAreaData(show: true, color: Colors.blue.withOpacity(0.3)),
            dotData: FlDotData(show: false),
          ),
        ],
      ),
    );
  }
}
