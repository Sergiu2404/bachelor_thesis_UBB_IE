class AppUser {
  final String uid;
  final int id;
  final String email;
  final String app_username;
  final double virtual_money_balance;

  AppUser({
    required this.uid,
    required this.id,
    required this.email,
    required this.app_username,
    required this.virtual_money_balance
  });

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'id': id,
      'email': email,
      'app_username': app_username,
      'virtual_money_balance': virtual_money_balance
    };
  }

  factory AppUser.fromMap(Map<String, dynamic> map) {
    return AppUser(
      uid: map['uid'],
      id: map['id'],
      email: map['email'],
      app_username: map['app_username'],
        virtual_money_balance: map['virtual_money_balance']
    );
  }

  // To print AppUser details (optional)
  @override
  String toString() {
    return 'AppUser{uid: $uid, id: $id, email: $email, app_username: $app_username}, virtual_money_balance: ${virtual_money_balance}';
  }
}
