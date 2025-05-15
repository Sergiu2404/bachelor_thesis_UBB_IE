try:
    from jose import jwt

    print("Successfully imported jose.jwt")
except ImportError as e:
    print(f"Import error: {e}")

import sys

print("\nPython path:")
for path in sys.path:
    print(path)