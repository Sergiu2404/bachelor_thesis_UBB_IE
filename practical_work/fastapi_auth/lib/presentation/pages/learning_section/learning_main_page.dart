import 'package:flutter/material.dart';

import '../../home_page.dart';

class LearningMainPage extends StatefulWidget {
  const LearningMainPage({super.key});

  @override
  State<LearningMainPage> createState() => _LearningMainPageState();
}

class _LearningMainPageState extends State<LearningMainPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text("Learning Center",
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pushReplacement(
                context, MaterialPageRoute(builder: (context) => const HomePage()));
          },
        ),
        backgroundColor: Colors.deepPurple,
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // HEADER
            const Text(
              "Welcome to Your Financial Educational App",
              style: TextStyle(
                  fontSize: 26, fontWeight: FontWeight.bold, color: Colors.black),
            ),
            const SizedBox(height: 10),

            // SUBHEADER
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.deepPurple.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text(
                "This is the Learning Section, the most important part of the app. Here, you'll gain crucial financial knowledge before making investment decisions.",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
              ),
            ),

            const SizedBox(height: 20),

            // MAIN CONTENT
            const Text(
              "Why Should You Learn?",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),

            const Text(
              "At some point while doing quizzes, you might wonder: "
                  '"Why should I learn about the South Sea Bubble Burst or the 2008 financial crisis?"\n\n'
                  "As an investor, you must make informed decisions. Understanding past financial events helps you avoid costly mistakes. Instead of losing real money, learn from history and apply structured strategies to your investments.",
              style: TextStyle(fontSize: 18, height: 1.5),
            ),

            const SizedBox(height: 20),

            // STRUCTURED LEARNING
            Container(
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.blueAccent.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "The Best Learning Path:",
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 10),
                  Text(
                    "1️⃣ Start by reading articles in this section.\n"
                        "2️⃣ Solve quizzes to reinforce your knowledge.\n"
                        "3️⃣ Experiment in the demo investing account.\n"
                        "4️⃣ Use the AI Stock Predictor (only after reaching \$50k in demo investing).",
                    style: TextStyle(fontSize: 18, height: 1.5),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // STRATEGY TIP
            const Text(
              "Strategic Advice",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),

            const Text(
              "If you start directly with the demo investing account, you may lose money quickly unless you're lucky. If you fail, you'll be required to complete quizzes to regain virtual money. However, solving quizzes without studying might waste your time.\n\n"
                  "🔹 **Balanced Learning Approach**:\n"
                  "✓ Read a few articles → Solve a quiz → Read more → Solve another quiz → Invest with virtual money.\n"
                  "✓ This structured approach ensures better decision-making in real-world investing.",
              style: TextStyle(fontSize: 18, height: 1.5),
            ),

            const SizedBox(height: 20),

            // FINAL NOTE
            Container(
              padding: const EdgeInsets.all(15),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Final Tip",
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  SizedBox(height: 10),
                  Text(
                    "Remember: The AI Predictor is costly. Use it only when you have over \$50k in demo investments to make it worthwhile.",
                    style: TextStyle(fontSize: 18, height: 1.5),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 30),

            // BUTTON TO START LEARNING
            Center(
              child: ElevatedButton(
                onPressed: () {
                  // Navigate to articles or quizzes
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: const Text(
                  "Start Learning Now",
                  style: TextStyle(fontSize: 18, color: Colors.white),
                ),
              ),
            ),

            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}
