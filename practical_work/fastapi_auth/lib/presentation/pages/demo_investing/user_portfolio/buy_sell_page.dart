import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:fastapi_auth/data/services/portfolio_api.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/user_portfolio/user_portfolio_list.dart';
import 'package:flutter/material.dart';

class BuySellPage extends StatefulWidget {
  final String action;
  final String symbol;
  final double currentPrice;

  const BuySellPage({
    super.key,
    required this.action,
    required this.symbol,
    required this.currentPrice,
  });

  @override
  _BuySellPageState createState() => _BuySellPageState();
}

class _BuySellPageState extends State<BuySellPage> {
  final PortfolioService _portfolioService = PortfolioService();
  final AuthService _authService = AuthService();

  Map<String, dynamic> currentConnectedUser = {};
  int _quantity = 1;
  bool _isProcessing = false;
  String _responseMessage = "";
  int ownedQuantity = 0;

  @override
  void initState() {
    super.initState();
    _loadUserData();
    _loadPortfolioData();
  }

  Future<void> _loadPortfolioData() async {
    final fetchedPortfolioCompanies = await _portfolioService.getPortfolioForUser();

    if(fetchedPortfolioCompanies.isNotEmpty){
      //search company with symbol in the fetched portfolio companies
      for(final  company in fetchedPortfolioCompanies){
        if(company.symbol == widget.symbol && company.username == currentConnectedUser["username"]) {
          ownedQuantity = company.quantity;
          break;
        }
      }
    }
    setState(() {});
  }

  Future<void> _loadUserData() async {
    currentConnectedUser = await _authService.getCurrentUser();
    setState(() {});
  }

  void _showConfirmationDialog() {
    double totalCost = _quantity * widget.currentPrice;
    double newBalance = widget.action == "Buy"
        ? (currentConnectedUser["virtual_money_balance"] ?? 0) - totalCost
        : (currentConnectedUser["virtual_money_balance"] ?? 0) + totalCost;

    if (widget.action == "Buy" && newBalance < 0) {
      _showErrorDialog("❌ Insufficient balance! You need \$${totalCost.toStringAsFixed(2)}.");
      return;
    }

    if (widget.action == "Sell" && (ownedQuantity == 0 || _quantity > ownedQuantity)) {
      _showErrorDialog("❌ You don't own enough shares to sell.");
      return;
    }

    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text("Confirm ${widget.action}"),
          content: Text(
            "Are you sure you want to ${widget.action.toLowerCase()} $_quantity stocks of ${widget.symbol} at \$${widget.currentPrice.toStringAsFixed(2)} each?\n\n"
                "💰 Updated Balance: \$${newBalance.toStringAsFixed(2)}",
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text("Cancel", style: TextStyle(color: Colors.red)),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                _performTransaction();
              },
              child: Text("Yes, ${widget.action}"),
            ),
          ],
        );
      },
    );
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text("Error", style: TextStyle(color: Colors.red)),
          content: Text(message),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text("OK"),
            ),
          ],
        );
      },
    );
  }

  Future<void> _performTransaction() async {
    setState(() {
      _isProcessing = true;
      _responseMessage = "";
    });

    Map<String, dynamic> result = widget.action == "Buy"
        ? await _portfolioService.buyStockForUser(widget.symbol, _quantity)
        : await _portfolioService.sellStockForUser(widget.symbol, _quantity);

    setState(() {
      _responseMessage = result.containsKey("error")
          ? "❌ ${result["error"]}"
          : "✅ ${widget.action} successful!";
      _isProcessing = false;
    });

    if (!result.containsKey("error")) {
      Navigator.of(context).pop(); // Close BuySellPage
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const UserPortfolioList(),
        ),
      );
    } else {
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text("Error"),
            content: Text(_responseMessage),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: Text("OK"),
              ),
            ],
          );
        },
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("${widget.action} Stock"),
        backgroundColor: widget.action == "Buy" ? Colors.lightBlue : Colors.red,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildInfoRow("Symbol:", widget.symbol),
            _buildInfoRow("Current Price:", "\$${widget.currentPrice.toStringAsFixed(2)}"),
            if (widget.action == "Sell") _buildInfoRow("Owned Quantity:", "$ownedQuantity"),
            const SizedBox(height: 16),
            Text("Quantity:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            TextFormField(
              keyboardType: TextInputType.number,
              initialValue: "1",
              onChanged: (value) {
                setState(() {
                  _quantity = int.tryParse(value) ?? 1;
                });
              },
              decoration: InputDecoration(
                border: OutlineInputBorder(),
                hintText: "Enter quantity",
              ),
            ),
            const SizedBox(height: 24),
            Center(
              child: ElevatedButton(
                onPressed: _isProcessing ? null : _showConfirmationDialog,
                child: _isProcessing
                    ? CircularProgressIndicator(color: Colors.white)
                    : Text(widget.action),
                style: ElevatedButton.styleFrom(
                  backgroundColor: widget.action == "Buy" ? Colors.lightBlue : Colors.red,
                  padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          Text(value, style: TextStyle(fontSize: 16)),
        ],
      ),
    );
  }
}
