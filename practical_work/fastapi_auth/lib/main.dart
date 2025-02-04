import 'package:fastapi_auth/home_page.dart';
import 'package:fastapi_auth/login_page.dart';
import 'package:fastapi_auth/registration_page.dart';
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
      initialRoute: '/register',
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