import 'package:fastapi_auth/core/utils/user_provider.dart';
import 'package:fastapi_auth/presentation/home_page.dart';
import 'package:fastapi_auth/presentation/pages/auth/login_page.dart';
import 'package:fastapi_auth/presentation/pages/auth/registration_page.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/demo_investing_main_page.dart';
import 'package:fastapi_auth/presentation/pages/demo_investing/user_portfolio/user_portfolio_list.dart';
import 'package:fastapi_auth/presentation/pages/learning_section/learning_main_page.dart';
import 'package:fastapi_auth/presentation/pages/quiz/quiz_page.dart';
import 'package:fastapi_auth/presentation/welcome_page.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    ChangeNotifierProvider(
        create: (_) => UserProvider(),
      child: const MyApp()
    )
      // const MyApp()
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: '/',
      routes: {
        '/': (context) => const WelcomePage(title: 'Welcome to the best financial app for any beginner'),
        '/register': (context) => RegisterScreen(),
        '/login': (context) => LoginScreen(),
        '/home': (context) => const HomePage(),
        '/demo-investing': (context) => const DemoInvestingPage(),
        '/user-portfolio': (context) => const UserPortfolioList(),
        '/quiz-page': (context) => const QuizPage(),
        'learning-page': (context) => const LearningMainPage()
      },
    );
  }
}
