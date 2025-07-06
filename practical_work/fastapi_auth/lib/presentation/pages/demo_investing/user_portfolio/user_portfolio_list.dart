import 'package:fastapi_auth/data/models/portfolio_company.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:fastapi_auth/data/services/portfolio_api.dart';
import 'package:fastapi_auth/data/services/stock_data_api.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/user_portfolio/portfolio_company_page.dart';
import 'package:flutter/material.dart';
import '../../../custom_widgets/portfolio_company_card.dart'; // Import the card
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
    
    _loadUserData();
    _fetchPortfolioStocks();
  }

  void _loadUserData() async {
    currentConnectedUser = await _authService.getCurrentUser();
    setState(() {});
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

  Future<void> _refreshPortfolio() async {
    setState(() {
      isLoading = true;
    });
    await _fetchPortfolioStocks();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Your Portfolio", style: TextStyle(color: Colors.white),),
        // actions: [
        //   Padding(
        //     padding: EdgeInsets.only(right: 20),
        //     child: Text(
        //       (currentConnectedUser["virtual_money_balance"] as num?)?.toStringAsFixed(2) ?? "0.00",
        //       style: TextStyle(color: Colors.white),
        //     )
        //
        //   )
        // ],
        backgroundColor: Colors.lightBlue,
      ),

      body: RefreshIndicator(
        onRefresh: _refreshPortfolio,
        child: isLoading
            ? const Center(child: CircularProgressIndicator())
            : stocks.isEmpty
            ? const Center(child: Text("Your portfolio is empty for now"))
            : Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "Total Portfolio Evaluation: \$${stocks.fold(0.0, (sum, stock) => sum + stock.totalCurrentValue).toStringAsFixed(2)}",
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              Text(
                "Number of Holdings: ${stocks.length}",
                style: const TextStyle(fontSize: 16, color: Colors.black54),
              ),
              const SizedBox(height: 8),
              Text(
                "Virtual Balance: \$${(currentConnectedUser["virtual_money_balance"] as num?)?.toStringAsFixed(2) ?? "0.00"}",
                style: const TextStyle(fontSize: 16, color: Colors.black54),
              ),
              const Divider(height: 30),
              Expanded(
                child: ListView.builder(
                  physics: const AlwaysScrollableScrollPhysics(),
                  itemCount: stocks.length,
                  itemBuilder: (context, index) {
                    final stock = stocks[index];
                    return GestureDetector(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => PortfolioCompanyPage(stock: stock),
                          ),
                        ).then((_) => _refreshPortfolio());
                      },
                      child: PortfolioCompanyCard(
                        symbol: stock.symbol,
                        companyName: stock.companyName,
                        currentPrice: stock.totalCurrentValue / stock.quantity,
                        quantity: stock.quantity,
                        averagePurchasePrice: stock.averageBuyPrice,
                        currentEvaluation: stock.totalCurrentValue,
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
