import yfinance as yf

# Download data for the S&P 500 index for the 10 trading days after May 23, 2025
gspc = yf.download("^GSPC", start="2025-05-24", end="2025-06-12")

print(gspc)
