import 'package:flutter/material.dart';

class PredictorPage extends StatefulWidget {
  const PredictorPage({super.key});

  @override
  State<PredictorPage> createState() => _PredictorPageState();
}

class _PredictorPageState extends State<PredictorPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Text("Stock Price Predictor"),
    );
  }
}
