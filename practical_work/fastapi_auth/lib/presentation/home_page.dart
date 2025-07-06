import 'package:fastapi_auth/presentation/pages/prediction_section/prediction_main_page.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:fastapi_auth/core/utils/user_provider.dart';
import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/demo_investing_main_page.dart';
import 'package:fastapi_auth/presentation/pages/learning_section/learning_main_page.dart';
import 'package:fastapi_auth/presentation/pages/quiz/quiz_page.dart';

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
    if (userData == null) {
      return const Center(child: Text("Welcome Guest"));
    }

    final balance = (userData!['virtual_money_balance'] as num).toDouble();
    final balanceText = "\$${balance.toStringAsFixed(2)}";

    String message = "Level ";
    Color messageColor;

    if (balance < 100) {
      message = "Bankrupt!";
      messageColor = Colors.red.shade700;
    } else if (balance < 1000) {
      message = "Danger Zone!";
      messageColor = Colors.orange.shade900;
    } else if (balance < 10000) {
      message = "Climbing the Ladder";
      messageColor = Colors.amber.shade500;
    } else if (balance < 100000) {
      message = "Future Financier";
      messageColor = Colors.blue.shade700;
    } else {
      message = "Master of Money!";
      messageColor = Colors.green.shade800;
    }

    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 32.0),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Center(
              child: Text(
              "Welcome to your financial teaching app, ${userData!['username'] ?? 'Guest'}",
              style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
              ),
            ),
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.symmetric(vertical: 20.0, horizontal: 24.0),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.1),
                    spreadRadius: 4,
                    blurRadius: 10,
                    offset: const Offset(0, 4),
                  ),
                ],
              ),
              child: Column(
                children: [
                  Text(
                    "Email: ${userData!['email']}",
                    style: TextStyle(fontSize: 16, color: Colors.grey[800]),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    "Balance: $balanceText",
                    style: TextStyle(
                      fontSize: 20,
                      color: Colors.teal[700],
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    message,
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: messageColor,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final userProvider = Provider.of<UserProvider>(context, listen: false);
    if (userProvider.userData == null && !userProvider.isLoading) {
      userProvider.setLoading(true);
      AuthService().getCurrentUser().then((user) {
        userProvider.setUser(user);
      });
    }
  }

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    final userProvider = Provider.of<UserProvider>(context);

    final List<Widget> _pages = [
      userProvider.isLoading
          ? const Center(child: CircularProgressIndicator())
          : HomeContent(userData: userProvider.userData),
      const LearningMainPage(),
      const QuizPage(),
      const DemoInvestingPage(),
      const PredictionMainPage(),
    ];

    return Scaffold(
      appBar: _selectedIndex == 0
          ? AppBar(
        backgroundColor: Colors.blue,
        automaticallyImplyLeading: false,
        title: const Text('Home Page', style: TextStyle(color: Colors.white),),
        actions: [
          // IconButton(
          //   icon: const Icon(Icons.exit_to_app),
          //   onPressed: () async {
          //     await AuthService().logout();
          //     Provider.of<UserProvider>(context, listen: false).clearUser();
          //     Navigator.pushNamedAndRemoveUntil(context, "/login", (_) => false);
          //   },
          // ),
        ],
      )
          : null,
      body: _pages[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        type: BottomNavigationBarType.fixed,
        items: const [
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
