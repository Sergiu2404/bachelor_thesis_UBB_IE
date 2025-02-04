import 'package:fastapi_auth/presentation/home_page.dart';
import 'package:fastapi_auth/presentation/pages/auth/login_page.dart';
import 'package:fastapi_auth/presentation/pages/auth/registration_page.dart';
import 'package:flutter/material.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  //await Firebase.initializeApp();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});


  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      initialRoute: '/home',
      routes: {
        '/register': (context) => RegisterScreen(),
        '/login': (context) => LoginScreen(),
        '/home': (context) => const HomePage(),
        // '/demo-investing': (context) => const DemoInvestingPage(),
        // '/user-portfolio': (context) => const UserPortfolioPage(),
      },
    );

  }
}