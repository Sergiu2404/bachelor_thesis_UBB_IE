class StockData{
  String companyName;
  String symbol;
  double price;

  StockData({required this.companyName, required this.symbol,required this.price });

  factory StockData.fromJson(Map<String, dynamic> json) {
    return StockData(
      symbol: json['symbol'] ?? '',
      companyName: json['company_name'] ?? '',
      price: (json['latest_price'] ?? 0.0).toDouble(),
    );
  }

  Map<String, dynamic> toJson(){
    return {
      "company_name": companyName,
      "symbol": symbol,
      "latest_price": price
    };
  }

  String toString(){
    return "company name: ${companyName}, symbol: ${symbol}, price: ${price.toString()}";
  }
}