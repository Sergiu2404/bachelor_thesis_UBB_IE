class PortfolioCompany {
  String username;
  String symbol;
  String companyName;
  int quantity;
  double averageBuyPrice;
  double totalCurrentValue;

  PortfolioCompany({
    required this.username,
    required this.symbol,
    required this.companyName,
    required this.quantity,
    required this.averageBuyPrice,
    required this.totalCurrentValue,
  });

  factory PortfolioCompany.fromJson(Map<String, dynamic> json) {
    return PortfolioCompany(
      username: json["username"] ?? "",
      symbol: json["symbol"] ?? "",
      companyName: json["company_name"] ?? "",
      quantity: (json["quantity"] ?? 0).toInt(),
      averageBuyPrice: (json["average_buy_price"] ?? 0).toDouble(),
      totalCurrentValue: (json["total_current_price"] ?? 0).toDouble(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      "username": username,
      "symbol": symbol,
      "company_name": companyName,
      "quantity": quantity,
      "average_buy_price": averageBuyPrice,
      "total_current_price": totalCurrentValue,
    };
  }

  @override
  String toString() {
    return "PortfolioCompany(username: $username, symbol: $symbol, companyName: $companyName, quantity: $quantity, averageBuyPrice: $averageBuyPrice, totalCurrentValue: $totalCurrentValue)";
  }
}
