# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from MakeTextfile import maketxt
from trainmodel import run_training
from generate_games import generate_games, generate_games_rand,generate_games_file
from perplexity import calc_perplexity


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #maketxt("jan2013.pgn")
    run_training('gpt2', '../data/jan2013.txt', '../data/london.txt')

    #startpositionen = ['e4 e5', 'e4 c5', 'e4 d5', 'd4 d5', 'c4 e5']
    #topp = 0.92
    #maxlen = 100
    #generate_games('gpt2', startpositionen, topp, maxlen, 3)
    #generate_games_rand('gpt2', topp, maxlen, 3, 5)
    #generate_games_file('gpt2', topp, maxlen, 3, 5, './data/jan2013.pgn', 2)

    #score = calc_perplexity('gpt2', '../data/jan2013_kurz2.txt')
    #print(score)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
