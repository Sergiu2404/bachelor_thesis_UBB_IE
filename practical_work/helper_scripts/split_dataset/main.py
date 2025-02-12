import pandas as pd

file_path = "Reviews.csv"
chunk_size = 50000  # rows per splitted file

df = pd.read_csv(file_path)

# Split df into smaller parts
for i, chunk in enumerate(range(0, len(df), chunk_size)):
    df_chunk = df.iloc[chunk:chunk + chunk_size]
    output_filename = f"Reviews{i + 1}.csv"
    df_chunk.to_csv(output_filename, index=False)
    print(f"Created: {output_filename}")

print("Splitting complete!")