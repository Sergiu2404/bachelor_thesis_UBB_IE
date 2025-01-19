import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

// Create a separate widget for home content
class HomeContent extends StatelessWidget {
  const HomeContent({super.key});

  @override
  Widget build(BuildContext context) {
    return const Center(
      child: Text('Home Content'),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  // Use HomeContent instead of HomePage
  final List<Widget> _pages = const [
    HomeContent(),
    //LearningPage(),
    //DemoInvestingPage(),
    //PredictorPage(),
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
              FirebaseAuth.instance.signOut();
              Navigator.pushNamed(context, "/login");
            },
          ),
        ],
      ),
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed, // Add this if you have more than 3 items
        items: const <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.school),
            label: 'Learning',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.money),
            label: 'Demo Investing',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.analytics),
            label: 'Predictor',
          ),
        ],
      ),
    );
  }
}

// Keep your other page widgets as they are...