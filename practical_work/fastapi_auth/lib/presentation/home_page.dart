// import 'dart:convert';
// import 'package:fastapi_auth/presentation/pages/demo_investing/demo_investing_main_page.dart';
// import 'package:fastapi_auth/presentation/pages/learning_section/learning_main_page.dart';
// import 'package:fastapi_auth/presentation/pages/quiz/quiz_page.dart';
// import 'package:flutter/material.dart';
// import 'package:fastapi_auth/data/services/auth_api.dart';
//
// class HomePage extends StatefulWidget {
//   const HomePage({super.key});
//
//   @override
//   State<HomePage> createState() => _HomePageState();
// }
//
// class HomeContent extends StatelessWidget {
//   final Map<String, dynamic>? userData;
//
//   const HomeContent({super.key, this.userData});
//
//   @override
//   Widget build(BuildContext context) {
//     return userData == null
//         ? const Center(child: CircularProgressIndicator())
//         : Center(
//       child: Padding(
//         padding: const EdgeInsets.symmetric(horizontal: 32.0),
//         child: Column(
//           mainAxisSize: MainAxisSize.min,
//           crossAxisAlignment: CrossAxisAlignment.center,
//           children: [
//             const Icon(Icons.account_circle, size: 100, color: Colors.blue),
//             const SizedBox(height: 10),
//             Text(
//               "Welcome, ${userData!['username'] ?? 'Guest'}",
//               style: const TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
//               textAlign: TextAlign.center,
//             ),
//             const SizedBox(height: 8),
//             Text("Email: ${userData!['email']}", style: const TextStyle(fontSize: 18), textAlign: TextAlign.center),
//             Text("Balance: \$${(userData!['virtual_money_balance'] as num).toStringAsFixed(2)}", style: const TextStyle(fontSize: 18), textAlign: TextAlign.center),
//             Text("Joined: ${userData!['created_at']}", style: const TextStyle(fontSize: 16), textAlign: TextAlign.center),
//           ],
//         ),
//       ),
//     );
//   }
// }
//
// class _HomePageState extends State<HomePage> {
//   int _selectedIndex = 0;
//   Map<String, dynamic>? userData;
//   bool isLoading = true;
//
//   @override
//   void initState() {
//     // TODO: implement initState
//     super.initState();
//     _fetchCurrentUser();
//   }
//
//   Future<void> _fetchCurrentUser() async {
//     try {
//       final user = await AuthService().getCurrentUser();
//       setState(() {
//         userData = user;
//         isLoading = false;
//       });
//     } catch (e) {
//       setState(() {
//         userData = null;
//         isLoading = false;
//       });
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text("Failed to load user: $e")),
//       );
//     }
//   }
//
//   void _onItemTapped(int index) {
//     setState(() {
//       _selectedIndex = index;
//     });
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     final List<Widget> _pages = [
//       isLoading
//           ? const Center(child: CircularProgressIndicator())
//           : HomeContent(userData: userData),
//       const LearningMainPage(),
//       QuizPage(userData: userData),
//       const DemoInvestingPage(),
//       const Center(child: Text("Predictor")),
//     ];
//
//     return Scaffold(
//       appBar: _selectedIndex == 0
//           ? AppBar(
//         automaticallyImplyLeading: false,
//         title: const Text('Home Page'),
//         actions: [
//           IconButton(
//             icon: const Icon(Icons.exit_to_app),
//             onPressed: () async {
//               await AuthService().logout();
//               Navigator.pushNamedAndRemoveUntil(context, "/login", (_) => false);
//             },
//           ),
//         ],
//       )
//           : null,
//       body: _pages[_selectedIndex],
//       bottomNavigationBar: BottomNavigationBar(
//         currentIndex: _selectedIndex,
//         onTap: _onItemTapped,
//         type: BottomNavigationBarType.fixed,
//         items: const [
//           BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
//           BottomNavigationBarItem(icon: Icon(Icons.school), label: 'Learning'),
//           BottomNavigationBarItem(icon: Icon(Icons.quiz), label: 'Quiz'),
//           BottomNavigationBarItem(icon: Icon(Icons.money), label: 'Demo Investing'),
//           BottomNavigationBarItem(icon: Icon(Icons.analytics), label: 'Predictor'),
//         ],
//       ),
//     );
//   }
// }







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
    return userData == null
        ? const Center(child: Text("Welcome Guest"))
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
            Text("Email: ${userData!['email']}", style: const TextStyle(fontSize: 18)),
            Text("Balance: \$${(userData!['virtual_money_balance'] as num).toStringAsFixed(2)}",
                style: const TextStyle(fontSize: 18)),
            Text("Joined: ${userData!['created_at']}", style: const TextStyle(fontSize: 16)),
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
      const Center(child: Text("Predictor")),
    ];

    return Scaffold(
      appBar: _selectedIndex == 0
          ? AppBar(
        automaticallyImplyLeading: false,
        title: const Text('Home Page'),
        actions: [
          IconButton(
            icon: const Icon(Icons.exit_to_app),
            onPressed: () async {
              await AuthService().logout();
              Provider.of<UserProvider>(context, listen: false).clearUser();
              Navigator.pushNamedAndRemoveUntil(context, "/login", (_) => false);
            },
          ),
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
