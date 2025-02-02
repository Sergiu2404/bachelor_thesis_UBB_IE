import 'package:flutter/material.dart';

import '../../../data/services/stocks_api/stocks_service_api.dart';

class DemoInvestingPage extends StatefulWidget {
  const DemoInvestingPage({super.key});

  @override
  State<DemoInvestingPage> createState() => _DemoInvestingPageState();
}

class _DemoInvestingPageState extends State<DemoInvestingPage> {
  final TextEditingController _searchController = TextEditingController();
  final StocksServiceAPI _stocksServiceAPI = StocksServiceAPI();
  late List<Map<String, dynamic>> stockResults = [];
  bool isLoading = false;
  String? errorMessage;

  @override
  void initState() {
    super.initState();
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
      final stocks = await _stocksServiceAPI.fetchStocksBySymbol(searchTerm);
      setState(() {
        stockResults = stocks;
        isLoading = false;
        errorMessage = stocks.isEmpty ? 'No matching stocks found' : null;
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
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Center(
              child: Text(
                'Balance: \$10,000',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: Colors.white,
                ),
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.account_balance),
            onPressed: () {
              Navigator.pushNamed(context, '/user-portfolio');
            },
            tooltip: 'View Portfolio',
          ),
        ],
      ),
      body: Column(
        children: [
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
          Expanded(
            child: isLoading
                ? const Center(child: CircularProgressIndicator())
                : stockResults.isEmpty && errorMessage == null
                ? const Center(
              child: Text('Search for stocks to see results'),
            )
                : ListView.builder(
              itemCount: stockResults.length,
              itemBuilder: (context, index) {
                final stock = stockResults[index];
                return Card(
                  margin: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 4,
                  ),
                  child: ListTile(
                    title: Row(
                      children: [
                        Text(
                          stock['symbol'] ?? stock['name'],
                          style: const TextStyle(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            stock['name'],
                            style: Theme.of(context).textTheme.bodyMedium,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                      ],
                    ),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const SizedBox(height: 4),
                        Text(
                          'Last Updated: ${stock['lastUpdated']}',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        if (stock['region'] != null)
                          Text(
                            'Region: ${stock['region']}',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                      ],
                    ),
                    trailing: Text(
                      '\$${stock['currentPrice'].toStringAsFixed(2)}',
                      style: Theme.of(context)
                          .textTheme
                          .titleMedium
                          ?.copyWith(
                        color: Colors.green,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
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