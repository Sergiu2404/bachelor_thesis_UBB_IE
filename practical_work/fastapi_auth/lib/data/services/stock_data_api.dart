import 'package:fastapi_auth/data/models/stock_data.dart';

class StockDataService{
  static const String baseUrl = 'http://10.0.2.2:8000';

  Future<StockData> getStockForSymbol(String symbol){
    return StockData(companyName: "", symbol: "", price: 0);
  }
}