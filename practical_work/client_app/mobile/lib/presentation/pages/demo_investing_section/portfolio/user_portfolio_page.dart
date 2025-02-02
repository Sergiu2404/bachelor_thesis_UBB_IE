import 'package:auth_firebase/presentation/pages/demo_investing_section/portfolio/user_portfolio_list.dart';
import 'package:flutter/material.dart';

class UserPortfolioPage extends StatefulWidget {
  const UserPortfolioPage({super.key});

  @override
  State<UserPortfolioPage> createState() => _UserPortfolioPageState();
}

class _UserPortfolioPageState extends State<UserPortfolioPage> {

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Your current portfolio"),
        backgroundColor: Colors.lightBlue,
      ),
      body: const Padding(
        padding: EdgeInsets.all(16.0),
        child: UserPortfolioList(), // Directly using the list
      ),
    );
  }
}
