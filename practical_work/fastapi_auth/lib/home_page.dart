import 'package:fastapi_auth/auth_api.dart';
import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  AuthService authService = AuthService();
  String? username;

  @override
  void initState() {
    super.initState();

    fetchCurrentUser();
  }

  Future<void> fetchCurrentUser() async {
    try{
      final userData = await authService.getCurrentUser();
      if(userData != null && userData.containsKey("username")){
        setState(() {
          username = userData["username"];
        });
      } else {
        setState(() {
          username = "Guest";
        });
      }
    } catch(error) {
      setState(() {
        username = "Guest";
      });

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: ${error.toString()}"))
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Welcome ${username ?? 'Loading...'}"),
      ),
      body: Center( child: Text("HOME PAGE")),
    );
  }
}
