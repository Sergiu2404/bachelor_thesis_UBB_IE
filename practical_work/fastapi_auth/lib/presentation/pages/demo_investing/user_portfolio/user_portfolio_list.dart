import 'package:fastapi_auth/data/models/portfolio_company.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:fastapi_auth/data/services/portfolio_api.dart';
import 'package:fastapi_auth/data/services/stock_data_api.dart';
import 'package:flutter/material.dart';
import 'portfolio_company_card.dart'; // Import the card
class UserPortfolioList extends StatefulWidget {
  const UserPortfolioList({super.key});

  @override
  State<UserPortfolioList> createState() => _UserPortfolioListState();
}

class _UserPortfolioListState extends State<UserPortfolioList> {
  final StockDataService _stockService = StockDataService();
  final AuthService _authService = AuthService();
  final PortfolioService _portfolioService = PortfolioService();
  Map<String, dynamic> currentConnectedUser = {};

  late List<PortfolioCompany> stocks = [];
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchPortfolioStocks();
  }

  Future<void> _fetchPortfolioStocks() async {
    try {
      final fetchedPortfolioCompanies = await _portfolioService.getPortfolioForUser();
      setState(() {
        if (fetchedPortfolioCompanies.isNotEmpty) stocks = fetchedPortfolioCompanies;
        isLoading = false;
      });
    } catch (error) {
      setState(() {
        isLoading = false;
      });
      print("Error fetching stocks: $error");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
          title: const Text("Your Portfolio"),
          backgroundColor: Colors.lightBlue,
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : stocks.isEmpty
          ? const Center(child: Text("Your portfolio is empty for now"))
          : Padding(
        padding: const EdgeInsets.all(16.0),
        child: ListView.builder(
          physics: AlwaysScrollableScrollPhysics(),
          itemCount: stocks.length,
          itemBuilder: (context, index) {
            final stock = stocks[index];
            return ListTile(
              title: Text(stock.companyName),
              subtitle: Text('Quantity: ${stock.quantity}'),
              trailing: Text('\$${stock.totalCurrentValue}'),
            );
          },
        ),
      ),
    );
  }

}
