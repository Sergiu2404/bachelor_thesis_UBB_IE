import 'package:auth_firebase/data/services/stocks_api/stocks_service_api.dart';
import 'package:flutter/material.dart';
import 'portfolio_company_card.dart'; // Import the card

class UserPortfolioList extends StatefulWidget {
  const UserPortfolioList({super.key});

  @override
  State<UserPortfolioList> createState() => _UserPortfolioListState();
}

class _UserPortfolioListState extends State<UserPortfolioList> {
  final StocksServiceAPI _stocksServiceAPI = StocksServiceAPI();
  late List<Map<String, dynamic>> stocks = [];
  bool isLoading = true;

  List<Map<String, dynamic>> _generateHardcodedStocks() {
    return List.generate(10, (i) => {
      'name': 'Stock $i',
      'currentPrice': (i + 1) * 10.0,
      'quantity': (i + 1) * 2,
      'purchasePrice': (i + 1) * 8.0,
      'totalEvaluation': (i + 1) * 10.0 * (i + 1) * 2,
    });
  }

  @override
  void initState() {
    super.initState();
    _fetchPortfolioStocks();
  }

  Future<void> _fetchPortfolioStocks() async {
    try {
      final fetchedPortfolioCompanies = await _stocksServiceAPI.fetchStockData();
      setState(() {
        stocks = fetchedPortfolioCompanies.isNotEmpty
            ? fetchedPortfolioCompanies
            : _generateHardcodedStocks();
        isLoading = false;
      });
    } catch (error) {
      setState(() {
        stocks = _generateHardcodedStocks();
        isLoading = false;
      });
      print("Error fetching stocks: $error");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView.builder(
          physics: AlwaysScrollableScrollPhysics(),
          itemCount: stocks.length,
          itemBuilder: (context, index) {
            final stock = stocks[index];
            return Padding(
              padding: const EdgeInsets.only(bottom: 12.0),
              child: PortfolioCompanyCard(
                companyName: stock['name'],
                currentPrice: stock['currentPrice'],
                quantity: 0,
                averagePurchasePrice: 0,
                currentEvaluation: 0,
              ),
            );
          },
        ),
      ),
    );
  }
}
