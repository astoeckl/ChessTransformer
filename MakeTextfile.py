import pandas as pd

def maketxt(file):
    df = pd.read_csv("./data/" + file)
    pgn_str = " <|endoftext|> ".join(df[df.columns[0]].tolist())

    filenames = file.split(".")

    with open ("./data/" + filenames[0] + ".txt", 'w') as f:
        f.write(pgn_str)