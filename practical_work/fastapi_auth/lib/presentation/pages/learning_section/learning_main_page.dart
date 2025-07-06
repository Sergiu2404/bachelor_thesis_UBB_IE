import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:url_launcher/url_launcher_string.dart';

import '../../home_page.dart';

class LearningMainPage extends StatefulWidget {
  const LearningMainPage({super.key});

  @override
  State<LearningMainPage> createState() => _LearningMainPageState();
}

class _LearningMainPageState extends State<LearningMainPage> {

  Future<void> _launchURL(String url) async {
    try {
      await launchUrlString(url);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text("Could not launch $url"),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Widget _buildLinkTile(String title, String url) {
    return InkWell(
      onTap: () => _launchURL(url),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6.0),
        child: Row(
          children: [
            const Icon(Icons.link, color: Colors.blue),
            const SizedBox(width: 8),
            Flexible(
              child: Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  color: Colors.blue,
                  decoration: TextDecoration.underline,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

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
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        elevation: 4,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              "Welcome to Your Financial Educational App",
              style: TextStyle(
                  fontSize: 26, fontWeight: FontWeight.bold, color: Colors.black),
            ),
            const SizedBox(height: 10),

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
                    "Start by reading / watching recommended materials in this section.\n"
                        "Solve quizzes to reinforce your knowledge.\n"
                        "Experiment in the demo investing account.\n"
                        "Use the AI Stock Predictor (be careful, it costs you \$1).",
                    style: TextStyle(fontSize: 18, height: 1.5),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            const Text(
              "Strategic Advice",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),

            const Text(
              "If you start directly with the demo investing account, you may lose money quickly unless you're lucky. If you fail, you'll be required to complete quizzes to regain virtual money. However, solving quizzes without studying might waste your time.\n\n"
                  "Balanced Learning Approach:\n"
                  "Read a few articles → Solve a quiz → Read more → Solve another quiz → Invest with virtual money.\n"
                  "This structured approach ensures better decision-making in real-world investing.",
              style: TextStyle(fontSize: 18, height: 1.5),
            ),

            const SizedBox(height: 20),

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
                    "Remember: The AI Predictor costs \$1. Don't rely on it too much:)",
                    style: TextStyle(fontSize: 18, height: 1.5),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            const Text(
              "Trusted Financial Learning Sources",
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),

            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildLinkTile("Mr. Money Mustache (Blogs)", "https://www.mrmoneymustache.com"),
                _buildLinkTile("Bogleheads Wiki", "https://www.bogleheads.org/wiki/Main_Page"),
                _buildLinkTile("Of Dollars and Data", "https://ofdollarsanddata.com"),
                _buildLinkTile("A Wealth of Common Sense", "https://awealthofcommonsense.com"),
                const SizedBox(height: 16),

                _buildLinkTile("Khan Academy - Finance & Capital Markets (Courses & Structured Learning)", "https://www.khanacademy.org/economics-finance-domain/core-finance"),
                _buildLinkTile("Coursera - Financial Markets (Yale)", "https://www.coursera.org/learn/financial-markets-global"),
                _buildLinkTile("Morningstar Investing Classroom", "https://www.morningstar.com/lp/investing-classroom"),
                const SizedBox(height: 16),

                _buildLinkTile("The Plain Bagel (YouTube)", "https://www.youtube.com/c/ThePlainBagel"),
                _buildLinkTile("Ben Felix", "https://www.youtube.com/@BenFelixCSI"),
                _buildLinkTile("Two Cents (PBS)", "https://www.youtube.com/c/TwoCentsPBS"),
                _buildLinkTile("Aswath Damodaran Lectures", "https://www.youtube.com/c/AswathDamodaranonValuation"),
              ],
            )

          ],
        ),
      ),
    );
  }
}