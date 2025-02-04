import 'dart:convert';

import 'package:fastapi_auth/data/services/auth_api.dart';
import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  AuthService authService = AuthService();
  //String? username;
  Map<String, dynamic>? userData;

  @override
  void initState() {
    super.initState();

    fetchCurrentUser();
  }

  Future<void> fetchCurrentUser() async {
    try{
      final user = await authService.getCurrentUser();
      setState(() {
        userData = user;
      });
    } catch(error) {
      setState(() {
        userData = null;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: ${error.toString()}"),)
      );
    }
  }

  // Future<void> fetchCurrentUser() async {
  //   try{
  //     final userData = await authService.getCurrentUser();
  //     if(userData != null && userData.containsKey("username")){
  //       setState(() {
  //         username = userData["username"];
  //       });
  //     } else {
  //       setState(() {
  //         username = "Guest";
  //       });
  //     }
  //   } catch(error) {
  //     setState(() {
  //       username = "Guest";
  //     });
  //
  //     ScaffoldMessenger.of(context).showSnackBar(
  //       SnackBar(content: Text("Error: ${error.toString()}"))
  //     );
  //   }
  // }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Welcome ${userData?['username'] ?? 'Guest'}"),
      ),
      body: userData == null
          ? const Center(child: CircularProgressIndicator())
          : Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text("Username: ${userData!['username']}", style: TextStyle(fontSize: 20)),
            Text("Email: ${userData!['email']}", style: TextStyle(fontSize: 18)),
            Text("Balance: \$${userData!['virtual_money_balance']}", style: TextStyle(fontSize: 18)),
            Text("Joined: ${userData!['created_at']}", style: TextStyle(fontSize: 16)),
          ],
        ),
      ),
    );
  }
}
