import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class HomeContent extends StatelessWidget {
  final Map<String, dynamic>? userData;

  const HomeContent({super.key, this.userData});

  @override
  Widget build(BuildContext context) {
    return userData == null
        ? const Center(child: CircularProgressIndicator())
        : Padding(
      padding: const EdgeInsets.all(16.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          const Icon(Icons.account_circle, size: 100, color: Colors.blue),
          const SizedBox(height: 10),
          Text(
            "Welcome, ${userData!['username'] ?? 'Guest'}",
            style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text("Email: ${userData!['email']}", style: const TextStyle(fontSize: 18)),
          Text("Balance: \$${userData!['virtual_money_balance']}", style: const TextStyle(fontSize: 18)),
          Text("Joined: ${userData!['created_at']}", style: const TextStyle(fontSize: 16)),
        ],
      ),
    );
  }
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;
  AuthService authService = AuthService();
  Map<String, dynamic>? userData;

  @override
  void initState() {
    super.initState();
    fetchCurrentUser();
  }

  Future<void> fetchCurrentUser() async {
    try {
      final user = await authService.getCurrentUser();
      setState(() {
        userData = user;
      });
    } catch (error) {
      setState(() {
        userData = null;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: ${error.toString()}")),
      );
    }
  }

  // ✅ Pages for Navigation
  final List<Widget> _pages = [
    const HomeContent(),  // Home Page (Displays User Info)
    const Center(child: Text("Learning Page")), // Replace with actual LearningPage()
    const Center(child: Text("Demo Investing")), // Replace with actual DemoInvestingPage()
    const Center(child: Text("Predictor")), // Replace with actual PredictorPage()
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Home Page'),
        actions: [
          IconButton(
            icon: const Icon(Icons.exit_to_app),
            onPressed: () {
              // Handle logout logic here
              Navigator.pushNamed(context, "/login");
            },
          ),
        ],
      ),
      body: _selectedIndex == 0
          ? HomeContent(userData: userData) // Display User Info on Home Page
          : _pages[_selectedIndex], // Switch pages using BottomNavigationBar
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.school), label: 'Learning'),
          BottomNavigationBarItem(icon: Icon(Icons.money), label: 'Demo Investing'),
          BottomNavigationBarItem(icon: Icon(Icons.analytics), label: 'Predictor'),
        ],
      ),
    );
  }
}
