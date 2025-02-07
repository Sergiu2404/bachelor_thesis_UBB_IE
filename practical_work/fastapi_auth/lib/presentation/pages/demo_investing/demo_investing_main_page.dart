import 'package:fastapi_auth/data/models/stock_data.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:fastapi_auth/data/services/stock_data_api.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/stocks/stock_details_page.dart';
import 'package:flutter/material.dart';

import '../../custom_widgets/stock_card.dart';


class DemoInvestingPage extends StatefulWidget {
  const DemoInvestingPage({super.key});

  @override
  State<DemoInvestingPage> createState() => _DemoInvestingPageState();
}

class _DemoInvestingPageState extends State<DemoInvestingPage> {
  final StockDataService _stockDataService = StockDataService();
  final AuthService _authService = AuthService();
  final TextEditingController _searchController = TextEditingController();

  Map<String, dynamic> currentConnectedUser = {};

  String providerAPI = "yahoo";
  final List<String> providerOptions = ["yahoo", "alpha"];

  late List<StockData> stockResults = [];
  bool isLoading = false;
  String? errorMessage;

  @override
  void initState() {
    super.initState();

    _loadUserData();
  }

  void _loadUserData() async {
    currentConnectedUser = await _authService.getCurrentUser();
    setState(() {}); //trigger rebuild with new user data
  }

  void _searchStock() async {
    final searchTerm = _searchController.text.trim();
    if (searchTerm.isEmpty) {
      setState(() {
        errorMessage = 'Please enter a stock symbol';
        stockResults = [];
      });
      return;
    }

    setState(() {
      isLoading = true;
      errorMessage = null;
    });

    try {
      // Use the selected provider to fetch stock data
      final stocks = await _stockDataService.getStocksForSymbolSubstring(providerAPI, searchTerm);
      setState(() {
        stockResults = stocks;
        isLoading = false;
        errorMessage = stocks.isEmpty ? 'No stocks matching this symbol found' : null;
      });
    } catch (error) {
      setState(() {
        isLoading = false;
        errorMessage = error.toString();
        stockResults = [];
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Demo Investing Account"),
        backgroundColor: Colors.lightBlue,
        actions: [
          IconButton(
            icon: const Icon(Icons.account_balance),
            onPressed: () {
              print("Navigating to Portfolio...");
              Navigator.pushNamed(context, '/user-portfolio');
            },
            tooltip: 'View Portfolio',
          )
        ],
      ),
      body: Column(
        children: [
          // User details panel at the top
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'User: ${currentConnectedUser['username']}',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                Text(
                  'Balance: \$${currentConnectedUser['virtual_money_balance']}',
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: Colors.green,
                  ),
                ),
              ],
            ),
          ),

          // Dropdown to select API provider (Yahoo or Alpha)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: DropdownButton<String>(
              value: providerAPI,
              onChanged: (String? newValue) {
                setState(() {
                  providerAPI = newValue!;
                });
              },
              items: providerOptions.map<DropdownMenuItem<String>>((String value) {
                return DropdownMenuItem<String>(
                  value: value,
                  child: Text(value.toUpperCase()),
                );
              }).toList(),
            ),
          ),

          // Search bar for stock symbols
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                TextField(
                  controller: _searchController,
                  decoration: InputDecoration(
                    labelText: 'Search Stocks',
                    hintText: 'Enter stock symbol (e.g., MS, AAPL)',
                    errorText: errorMessage,
                    prefixIcon: const Icon(Icons.search),
                    suffixIcon: _searchController.text.isNotEmpty
                        ? IconButton(
                      icon: const Icon(Icons.clear),
                      onPressed: () {
                        _searchController.clear();
                        setState(() {
                          stockResults = [];
                          errorMessage = null;
                        });
                      },
                    )
                        : null,
                    border: const OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => _searchStock(),
                  textInputAction: TextInputAction.search,
                ),
                const SizedBox(height: 8),
                if (stockResults.isNotEmpty)
                  Padding(
                    padding: const EdgeInsets.only(left: 4),
                    child: Text(
                      '${stockResults.length} matching stocks found',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ),
              ],
            ),
          ),

          // Display search results
          Expanded(
            child: isLoading
                ? const Center(child: CircularProgressIndicator())
                : stockResults.isEmpty && errorMessage == null
                ? const Center(child: Text('Search for stocks to see results'))
                : ListView.builder(
              itemCount: stockResults.length,
              itemBuilder: (context, index) {
                final stock = stockResults[index];
                return GestureDetector(
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => StockDetailsPage(stockData: stock),
                      ),
                    );
                  },
                  child: StockCard(
                    companyName: stock.companyName ?? "Company Name",
                    symbol: stock.symbol,
                    price: stock.price,
                  ),
                );
              },
            ),
          ),

        ],
      ),
    );
  }
}
