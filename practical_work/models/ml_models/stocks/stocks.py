import yfinance as yf

stock = yf.Ticker("TTS.RO")
sector = stock.info.get('sector')
industry = stock.info.get('industry')
print(f"Sector: {sector}")
print(f"Industry: {industry}")