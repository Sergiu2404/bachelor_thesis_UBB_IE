class StockData{
  String companyName;
  String symbol;
  double price;

  StockData({required this.companyName, required this.symbol,required this.price });

  String toString(){
    return "company name: ${companyName}, symbol: ${symbol}, price: ${price.toString()}";
  }
}