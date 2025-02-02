import 'package:auth_firebase/presentation/pages/auth/login_page.dart';
import 'package:auth_firebase/presentation/pages/auth/registration_page.dart';
import 'package:auth_firebase/presentation/pages/demo_investing_section/demo_investing_main_page.dart';
import 'package:auth_firebase/presentation/pages/demo_investing_section/portfolio/user_portfolio_page.dart';
import 'package:auth_firebase/presentation/pages/home_page.dart';
import 'package:auth_firebase/presentation/pages/welcome_page.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      initialRoute: '/demo-investing',
      routes: {
        '/': (context) => const WelcomePage(title: 'Welcome to the best financial app for any beginner',),
        '/login': (context) => const LoginPage(),
        '/register': (context) => const RegistrationPage(),
        '/home': (context) => const HomePage(),
        '/demo-investing': (context) => const DemoInvestingPage(),
        '/user-portfolio': (context) => const UserPortfolioPage(),
      },
    );

  }
}