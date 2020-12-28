from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
import io
import chess.pgn
from datetime import date
import statistics
import random
from random import randrange

def savepgn(df,model_type,topp,maxlen,eval_type):
    # Korrekten Teil als PGN speichern
    for i in range(df.shape[0]):
        gamestr = str(df.iloc[i][0])
        pgn = io.StringIO(gamestr)
        game = chess.pgn.read_game(pgn)
        game.headers["Event"] = "Generierte Züge"
        game.headers["Site"] = "Hagenberg"
        game.headers["Date"] = str(date.today())
        game.headers["Round"] = str(i)
        game.headers["White"] = str(model_type)
        game.headers["Black"] = str(model_type)
        filenamepgn = '../data/games/partien_'+ eval_type + model_type + str(int(topp*100)) + str(maxlen) + '.pgn'
        print(game, file=open(filenamepgn, "a"), end="\n\n")


def count_moves(df):
    erglist = []

    for san in df['Zuege']:
        partie = san.split()
        board = chess.Board()

        i = 0
        for zug in partie:
            try:
                board.push_san(zug)
                i = i + 1
            except:
                erglist.append(i)
                break
    return (erglist,statistics.mean(erglist))


def generate_games(model_type, startpositionen, topp, maxlen, games_per_position):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelWithLMHead.from_pretrained("../data/models"+model_type)

    outputlist = []
    for start in startpositionen:
        input_ids = tokenizer.encode(start, return_tensors='pt')

        # generate text until the output length (which includes the context length) reaches max_length
        for i in range(games_per_position):
            output = model.generate(input_ids,
                                    max_length=maxlen,
                                    top_p=topp,
                                    do_sample=True)
            zuege = tokenizer.decode(output[0], skip_special_tokens=True)
            outputlist.append(zuege)

    df = pd.DataFrame(outputlist, columns=["Zuege"])
    filename = '../data/games/partien_'+model_type+str(int(topp*100))+str(maxlen)+'.csv'
    df.to_csv(filename, index=False)
    savepgn(df, model_type, topp, maxlen,"")

    # Wie viele Züge sind korrekt pro Partie / Durchschnitt
    # Von Startpositionen
    counts = count_moves(df)
    print(counts)


def random_move(board):
    move = random.choice(list(board.legal_moves))
    return board.san(move)


def generate_games_rand(model_type, topp, maxlen, games_per_position, depth):
    # Züege ab zufälliger Position generieren - Tiefe
    # Generate random Position
    gamestring = ""
    board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        move = random.choice(list(board.legal_moves))
        gamestring = gamestring + " " + board.san(move)
        board.push(move)

    gamelist = gamestring.split()
    start = ' '.join(gamelist[:depth])

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelWithLMHead.from_pretrained("../data/models" + model_type)

    outputlist = []
    input_ids = tokenizer.encode(start, return_tensors='pt')

    # generate text until the output length (which includes the context length) reaches max_length
    for i in range(games_per_position):
        output = model.generate(input_ids,
                                max_length=maxlen,
                                top_p=topp,
                                do_sample=True)
        zuege = tokenizer.decode(output[0], skip_special_tokens=True)
        outputlist.append(zuege)

    df = pd.DataFrame(outputlist, columns=["Zuege"])
    filename = '../data/games/partien_rand_' + model_type + str(int(topp * 100)) + str(maxlen) + '.csv'
    df.to_csv(filename, index=False)
    savepgn(df, model_type, topp, maxlen,"rand")

    counts = count_moves(df)
    print(counts)

# Zug ab zufälliger Position aus den Trainingsdaten generieren
# Generate random Position from Games
def generate_games_file(model_type, topp, maxlen, games_per_position, depth, file, position_per_file):
    df = pd.read_csv(file)

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelWithLMHead.from_pretrained("../data/models" + model_type)

    df = pd.read_csv(file)
    outputlist = []

    for i in range(position_per_file):
        # Generate random Position from Games
        gamestring = str(df.iloc[randrange(df.shape[0])][0])
        gamelist = gamestring.split()
        start = ' '.join(gamelist[:depth])

        # encode context the generation is conditioned on
        input_ids = tokenizer.encode(start, return_tensors='pt')

        # generate text until the output length (which includes the context length) reaches maxlength
        for i in range(games_per_position):
            output = model.generate(input_ids,
                                    max_length=maxlen,
                                    top_p=topp,
                                    do_sample=True)
            zuege = tokenizer.decode(output[0], skip_special_tokens=True)
            outputlist.append(zuege)

    df = pd.DataFrame(outputlist, columns=["Zuege"])
    filename = '../data/games/partien_file_' + model_type + str(int(topp * 100)) + str(maxlen) + '.csv'
    df.to_csv(filename, index=False)
    savepgn(df, model_type, topp, maxlen,"file")

    counts = count_moves(df)
    print(counts)