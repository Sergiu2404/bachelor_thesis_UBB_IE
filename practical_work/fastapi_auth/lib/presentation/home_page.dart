import 'dart:convert';
import 'package:fastapi_auth/presentation/pages/demo_investing/demo_investing_main_page.dart';
import 'package:fastapi_auth/presentation/pages/learning_section/learning_main_page.dart';
import 'package:fastapi_auth/presentation/pages/quiz/quiz_page.dart';
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
        : Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            const Icon(Icons.account_circle, size: 100, color: Colors.blue),
            const SizedBox(height: 10),
            Text(
              "Welcome, ${userData!['username'] ?? 'Guest'}",
              style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text("Email: ${userData!['email']}", style: const TextStyle(fontSize: 18), textAlign: TextAlign.center),
            Text("Balance: \$${userData!['virtual_money_balance']}", style: const TextStyle(fontSize: 18), textAlign: TextAlign.center),
            Text("Joined: ${userData!['created_at']}", style: const TextStyle(fontSize: 16), textAlign: TextAlign.center),
          ],
        ),
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
    // Fetch user data for the home screen
    _fetchCurrentUser();
  }

  Future<void> _fetchCurrentUser() async {
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

  final List<Widget> _pages = [
    const HomeContent(),  // Home content page
    const LearningMainPage(),
    const QuizPage(),
    const DemoInvestingPage(),  // Demo investing page
    const Center(child: Text("Predictor")),  // Placeholder page
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _selectedIndex == 0
          ? AppBar(
        automaticallyImplyLeading: false,
        title: const Text('Home Page'),
        actions: [
          IconButton(
            icon: const Icon(Icons.exit_to_app),
            onPressed: () async {
              AuthService authService = AuthService();
              await authService.logout();
              Navigator.pushNamedAndRemoveUntil(context, "/login", (route) => false);
              //Navigator.pushNamed(context, "/login");
            },
          ),
        ],
      )
          : null,  // No AppBar for other pages

      body: _selectedIndex == 0
          ? HomeContent(userData: userData)  // Show HomeContent on HomePage
          : _pages[_selectedIndex],  // Show other pages for different index

      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.school), label: 'Learning'),
          BottomNavigationBarItem(icon: Icon(Icons.quiz), label: 'Quiz'),
          BottomNavigationBarItem(icon: Icon(Icons.money), label: 'Demo Investing'),
          BottomNavigationBarItem(icon: Icon(Icons.analytics), label: 'Predictor'),
        ],
      ),
    );
  }
}
