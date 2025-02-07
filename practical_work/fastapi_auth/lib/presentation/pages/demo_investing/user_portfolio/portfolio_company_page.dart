import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../../../data/models/portfolio_company.dart';
import '../../../../data/services/stock_data_api.dart';
import 'buy_sell_page.dart';

class PortfolioCompanyPage extends StatefulWidget {
  final PortfolioCompany stock;

  const PortfolioCompanyPage({super.key, required this.stock});

  @override
  _PortfolioCompanyPageState createState() => _PortfolioCompanyPageState();
}

class _PortfolioCompanyPageState extends State<PortfolioCompanyPage> {
  final StockDataService _stockService = StockDataService();
  Map<String, double> monthlyPrices = {};
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchStockDetails();
  }

  Future<void> _fetchStockDetails() async {
    try {
      final monthly = await _stockService.getMonthlyStockData("yahoo", widget.stock.symbol);
      setState(() {
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
      appBar: AppBar(
        title: Text(widget.stock.companyName),
        backgroundColor: Colors.blueAccent,
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Symbol: ${widget.stock.symbol}",
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text("Current Price: \$${(widget.stock.totalCurrentValue / widget.stock.quantity).toStringAsFixed(2)}"),
            Text("Quantity: ${widget.stock.quantity}"),
            Text("Average Purchase Price: \$${widget.stock.averageBuyPrice.toStringAsFixed(2)}"),
            Text("Total Evaluation: \$${widget.stock.totalCurrentValue.toStringAsFixed(2)}"),
            const SizedBox(height: 20),

            // Chart Display
            if (monthlyPrices.isNotEmpty)
              SizedBox(height: 300, child: _buildStockChart()),

            const SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Buy Button
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => BuySellPage(
                          action: "Buy",
                          symbol: widget.stock.symbol,
                          currentPrice: widget.stock.totalCurrentValue / widget.stock.quantity,
                        ),
                      ),
                    );
                  },
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
                  child: const Text("Buy", style: TextStyle(fontSize: 18)),
                ),
                const SizedBox(width: 20),

                // Sell Button
                ElevatedButton(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => BuySellPage(
                          action: "Sell",
                          symbol: widget.stock.symbol,
                          currentPrice: widget.stock.totalCurrentValue / widget.stock.quantity,
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