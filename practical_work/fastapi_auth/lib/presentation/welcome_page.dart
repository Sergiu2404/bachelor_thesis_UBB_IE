import 'package:flutter/material.dart';

class WelcomePage extends StatefulWidget {
  final String title;
  const WelcomePage({super.key, required this.title});

  @override
  State<WelcomePage> createState() => _WelcomePageState();
}

class _WelcomePageState extends State<WelcomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            colors: [
              Color(0xFF0D47A1), // deep blue
              Color(0xFF1976D2), // light blue
              Color(0xFF42A5F5), // skyblue
            ],
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
          const Text(
          'Welcome to FinLearn!',
          style: TextStyle(
            fontSize: 36,
            fontWeight: FontWeight.bold,
            color: Colors.white,
          ),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 20),
        const Text(
          'Your app for stock price prediction, virtual investment and financial education.',
        style: TextStyle(
        fontSize: 18,
          color: Colors.white70,
        ),
        textAlign: TextAlign.center,
      ),
      const SizedBox(height: 40),
      Image.asset(
        'assets/images/fin_welcome.jpg', // Replace with your image asset
        height: 350,
        width: 350,
      ),
      const SizedBox(height: 40),
      ElevatedButton(
        onPressed: () {
          Navigator.pushNamedAndRemoveUntil(context, '/login', (route) => false);
        },
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
          backgroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
          ),
        ),
        child: const Text(
          'Already have an account? Log In',
          style: TextStyle(
            color: Color(0xFF0D47A1),
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      const SizedBox(height: 20),
      ElevatedButton(
        onPressed: () {
          Navigator.pushNamedAndRemoveUntil(context, '/register', (route) => false);
        },
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
          backgroundColor: Colors.orange,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(30),
          ),
        ),
        child: const Text(
          'Create an Account',
          style: TextStyle(
            color: Colors.white,
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      ],
    ),
    ),
    );
  }
}
