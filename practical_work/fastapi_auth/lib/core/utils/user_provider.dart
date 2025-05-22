import 'package:flutter/material.dart';

class UserProvider extends ChangeNotifier {
  Map<String, dynamic>? _userData;
  bool _isLoading = false;

  Map<String, dynamic>? get userData => _userData;
  bool get isLoading => _isLoading;

  void setUser(Map<String, dynamic> user) {
    _userData = user;
    _isLoading = false;
    notifyListeners();
  }

  void setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }

  void clearUser() {
    _userData = null;
    _isLoading = false;
    notifyListeners();
  }

  void updateBalance(double newBalance) {
    if (_userData != null) {
      _userData!['virtual_money_balance'] = newBalance;
      notifyListeners();
    }
  }
}