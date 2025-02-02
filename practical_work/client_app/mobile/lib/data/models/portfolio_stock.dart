// CREATE TABLE portfolio (
// id INTEGER PRIMARY KEY AUTOINCREMENT,
// user_id INTEGER,
// company_name TEXT,
// ticker_symbol TEXT,
// quantity INTEGER,
// purchase_price REAL,
// FOREIGN KEY (user_id) REFERENCES users (id)
// );


class PortfolioStock {
  final int id;
  final String user_id;
  final String company_name;
  final int quantity;
  final double average_purchase_price;

  PortfolioStock({
    required this.id,
    required this.user_id,
    required this.company_name,
    required this.quantity,
    required this.average_purchase_price
});

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'user_id': user_id,
      'company_name': company_name,
      'quantity': quantity,
      'average_purchase_price': average_purchase_price
    };
  }

  factory PortfolioStock.fromMap(Map<String, dynamic> map) {
    return PortfolioStock(
        id: map['id'],
        user_id: map['user_id'],
        company_name: map['company_name'],
        quantity: map['quantity'],
        average_purchase_price: map['average_purchase_price']
    );
  }

  @override
  String toString() {
    return 'PortfolioStock{id: $id, user_id: $user_id, company_name: $company_name, quantity: $quantity}, purchase_price: ${average_purchase_price}';
  }
}