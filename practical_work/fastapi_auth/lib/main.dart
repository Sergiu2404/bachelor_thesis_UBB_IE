import 'package:fastapi_auth/presentation/home_page.dart';
import 'package:fastapi_auth/presentation/pages/auth/login_page.dart';
import 'package:fastapi_auth/presentation/pages/auth/registration_page.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/demo_investing_main_page.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/user_portfolio/user_portfolio_list.dart';
import 'package:fastapi_auth/presentation/welcome_page.dart';
import 'package:flutter/material.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      initialRoute: '/home',  // Set the initial route here
      routes: {
        '/': (context) => const WelcomePage(title: 'Welcome to the best financial app for any beginner'),
        '/register': (context) => RegisterScreen(),
        '/login': (context) => LoginScreen(),
        '/home': (context) => const HomePage(),
        '/demo-investing': (context) => const DemoInvestingPage(),
        '/user-portfolio': (context) => const UserPortfolioList(),
      },
    );
  }
}
