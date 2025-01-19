class AppUser {
  final String uid; // Firebase Auth UID
  final int id; // Custom field for AppUser ID
  final String email; // AppUser email
  final String AppUsername; // AppUser AppUsername

  // Constructor
  AppUser({
    required this.uid,
    required this.id,
    required this.email,
    required this.AppUsername,
  });

  // Convert AppUser to a map to store in Firestore
  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'id': id,
      'email': email,
      'AppUsername': AppUsername,
    };
  }

  // Factory method to create a AppUser from Firestore data
  factory AppUser.fromMap(Map<String, dynamic> map) {
    return AppUser(
      uid: map['uid'],
      id: map['id'],
      email: map['email'],
      AppUsername: map['AppUsername'],
    );
  }

  // To print AppUser details (optional)
  @override
  String toString() {
    return 'AppUser{uid: $uid, id: $id, email: $email, AppUsername: $AppUsername}';
  }
}
