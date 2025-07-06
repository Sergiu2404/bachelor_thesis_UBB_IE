import 'package:fastapi_auth/presentation/pages/quiz/quiz_question_page.dart';
import 'package:flutter/material.dart';
import '../../../data/services/auth_api.dart';

class QuizPage extends StatefulWidget {
  final Map<String, dynamic>? userData;
  const QuizPage({super.key, this.userData});

  @override
  State<QuizPage> createState() => _QuizPageState();
}

class _QuizPageState extends State<QuizPage> {
  Map<String, dynamic>? currentConnectedUser;
  bool isLoading = true;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments;

    if (args is Map<String, dynamic>) {
      currentConnectedUser = args;
      isLoading = false;
    } else {
      _fetchCurrentUser();
    }
  }

  @override
  void initState() {
    super.initState();
    if (widget.userData != null) {
      currentConnectedUser = widget.userData;
      isLoading = false;
    } else {
      _fetchCurrentUser(); // fallback
    }
  }

  Future<void> _fetchCurrentUser() async {
    try {
      final user = await AuthService().getCurrentUser();
      setState(() {
        currentConnectedUser = user;
        isLoading = false;
      });
    } catch (_) {
      setState(() {
        currentConnectedUser = null;
        isLoading = false;
      });
    }
  }

  void _startQuiz(String difficulty, int reward) {
    try {
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) =>
              QuizQuestionPage(difficulty: difficulty, reward: reward),
        ),
      );
    } catch(e){
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    if (isLoading) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final isAuthenticated = currentConnectedUser != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text("Finance Quiz"),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pushReplacementNamed(context, '/home');
          },
        ),
      ),

      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blueAccent),
                ),
                child: const Text(
                  "This is the place where you can test your financial literacy. Each category consists of 10 multiple-choice questions. To receive the reward, you must answer at least 9 out of 10 correctly.",
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.w500),
                ),
              ),
              const SizedBox(height: 30),
              ElevatedButton(
                onPressed: () => _startQuiz("easy", 25),
                child: const Text("Start Easy Quiz (\$25 Reward)"),
              ),
              ElevatedButton(
                onPressed: () => _startQuiz("medium", 40),
                child: const Text("Start Medium Quiz (\$40 Reward)"),
              ),
              ElevatedButton(
                onPressed: () => _startQuiz("hard", 75),
                child: const Text("Start Hard Quiz (\$75 Reward)"),
              ),
            ],
          ),
        ),
      ),

    );
  }
}
