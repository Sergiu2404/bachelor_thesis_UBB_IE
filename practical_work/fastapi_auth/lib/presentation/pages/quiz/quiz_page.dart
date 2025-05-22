// import 'package:fastapi_auth/presentation/pages/quiz/quiz_question_page.dart';
// import 'package:flutter/material.dart';
// import '../../../data/services/auth_api.dart';
//
// class QuizPage extends StatefulWidget {
//   const QuizPage({super.key});
//
//   @override
//   State<QuizPage> createState() => _QuizPageState();
// }
//
// class _QuizPageState extends State<QuizPage> {
//   final AuthService _authService = AuthService();
//   Map<String, dynamic> currentConnectedUser = {};
//
//   @override
//   void initState() {
//     super.initState();
//     _loadUserData();
//   }
//
//   void _loadUserData() async {
//     currentConnectedUser = await _authService.getCurrentUser();
//     setState(() {});
//   }
//
//   void _startQuiz(String difficulty, int reward) {
//     Navigator.push(
//       context,
//       MaterialPageRoute(
//         builder: (context) => QuizQuestionPage(difficulty: difficulty, reward: reward),
//       ),
//     );
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     bool isAuthenticated = currentConnectedUser.isNotEmpty;
//
//     return Scaffold(
//       appBar: AppBar(
//           title: Text("Finance Quiz"),
//           leading: IconButton(
//             icon: Icon(Icons.arrow_back),
//             onPressed: () {
//               Navigator.pushReplacementNamed(context, '/home');
//             },
//     ),
//       ),
//       body: Center(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             Text(
//               isAuthenticated ? "Welcome ${currentConnectedUser['username']}" : "Play for Free!",
//               style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
//             ),
//             const SizedBox(height: 20),
//             ElevatedButton(
//               onPressed: () => _startQuiz("easy", 25),
//               child: Text("Start Easy Quiz (\$25 Reward)"),
//             ),
//             ElevatedButton(
//               onPressed: () => _startQuiz("medium", 40),
//               child: Text("Start Medium Quiz (\$40 Reward)"),
//             ),
//             ElevatedButton(
//               onPressed: () => _startQuiz("hard", 75),
//               child: Text("Start Hard Quiz (\$75 Reward)"),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }






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
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              isAuthenticated
                  ? "Welcome ${currentConnectedUser!['username']}"
                  : "Play as Guest",
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 20),
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
    );
  }
}
