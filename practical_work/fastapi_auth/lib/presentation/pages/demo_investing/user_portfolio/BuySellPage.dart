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
  final _formKey = GlobalKey<FormState>();
  int _quantity = 1;

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
                onPressed: () {
                  if (_formKey.currentState!.validate()) {
                    _formKey.currentState!.save();
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text("${widget.action}ing $_quantity shares of ${widget.symbol}")),
                    );
                  }
                },
                child: Text(widget.action),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
