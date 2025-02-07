import 'package:fastapi_auth/data/services/portfolio_api.dart';
import 'package:flutter/material.dart';

class BuySellPage extends StatefulWidget {
  final String action;  // "Buy" or "Sell"
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
  PortfolioService _portfolioService = PortfolioService();

  final _formKey = GlobalKey<FormState>();
  int _quantity = 1;
  bool _isProcessing = false;
  String _responseMessage = "";

  @override
  void initState() {
    super.initState();
  }

  Future<void> _performAction() async {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();

      setState(() {
        _isProcessing = true;
        _responseMessage = "";
      });

      Map<String, dynamic> result;

      if (widget.action.toLowerCase() == "buy") {
        result = await _portfolioService.buyStockForUser(widget.symbol, _quantity);
      } else if (widget.action.toLowerCase() == "sell") {
        result = await _portfolioService.sellStockForUser(widget.symbol, _quantity);
      } else {
        setState(() {
          _responseMessage = "Unknown action: ${widget.action}";
          _isProcessing = false;
        });
        return;
      }

      setState(() {
        if (result.containsKey("error")) {
          _responseMessage = result["error"];
        } else {
          _responseMessage = "${widget.action} action successful!";
        }
        _isProcessing = false;
      });

      // Show snackbar with the result
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(_responseMessage)),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("${widget.action} Stock")),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text("Symbol:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              Text(widget.symbol, style: TextStyle(fontSize: 16)),
              const SizedBox(height: 16),
              Text("Current Price:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              Text("\$${widget.currentPrice.toStringAsFixed(2)}", style: TextStyle(fontSize: 16)),
              const SizedBox(height: 16),
              Text("Quantity:", style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              TextFormField(
                keyboardType: TextInputType.number,
                initialValue: "1",
                validator: (value) {
                  if (value == null || int.tryParse(value) == null || int.parse(value) <= 0) {
                    return "Enter a valid quantity greater than 0";
                  }
                  return null;
                },
                onSaved: (value) {
                  _quantity = int.parse(value!);
                },
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _isProcessing ? null : _performAction,
                child: _isProcessing
                    ? CircularProgressIndicator(color: Colors.white)
                    : Text(widget.action),
              ),
              const SizedBox(height: 16),
              if (_responseMessage.isNotEmpty)
                Text(
                  _responseMessage,
                  style: TextStyle(color: _responseMessage.startsWith('Error') ? Colors.red : Colors.green),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
