class AppUser {
  final String uid;
  final int id;
  final String email;
  final String appUsername;

  AppUser({
    required this.uid,
    required this.id,
    required this.email,
    required this.appUsername,
  });

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'id': id,
      'email': email,
      'appUsername': appUsername,
    };
  }

  factory AppUser.fromMap(Map<String, dynamic> map) {
    return AppUser(
      uid: map['uid'],
      id: map['id'],
      email: map['email'],
      appUsername: map['appUsername'],
    );
  }

  @override
  String toString() {
    return 'AppUser{uid: $uid, id: $id, email: $email, appUsername: $appUsername}';
  }
}
