import 'dart:developer';

import 'package:firebase_auth/firebase_auth.dart';

class FirebaseAuthService{
  final FirebaseAuth _firebaseAuth = FirebaseAuth.instance;
  
  Future<User?> signUpWithEmailAndPassword(String email, String password) async{
    try{
      UserCredential credential = await _firebaseAuth.createUserWithEmailAndPassword(email: email, password: password);
      return credential.user;
    }catch(exception)
    {
      log("User signup: Some error occurred");
    }
    return null;
  }

  Future<User?> signInWithEmailAndPassword(String email, String password) async{
    try{
      UserCredential credential = await _firebaseAuth.signInWithEmailAndPassword(email: email, password: password);
      return credential.user;
    }catch(exception)
    {
      log("User signin: Some error occurred");
    }
    return null;
  }
}