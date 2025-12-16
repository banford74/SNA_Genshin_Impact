import os
import numpy as np
import pandas as pd
import logging
from argparse import ArgumentParser
from transformers import BertTokenizer

from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataPath', help='Path to the CSV data file with Japanese texts')
parser.add_argument('--lexPath', help='Path to the Japanese lexicon CSV file')
parser.add_argument('--lexNames', nargs="*", type=str, help='Names of the lexicons/column names in the lexicon CSV')
parser.add_argument('--savePath', help='Path to the save folder')

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

tqdm.pandas()


def read_lexicon(path, LexNames):
    try:
        df = pd.read_csv(path, encoding='utf-8')  # Specify UTF-8 encoding for Japanese text
        df = df.dropna(subset=['word'])  # Remove rows with missing words
        return df
    except FileNotFoundError:
        logging.error(f"Lexicon file not found: {path}")
        raise  # Re-raise the exception to stop execution


def prep_dim_lexicon(df, dim):
    ldf = df[['word', dim]]
    ldf = ldf.dropna(subset=[dim])
    ldf.drop_duplicates(subset=['word'], keep='first', inplace=True)
    ldf[dim] = ldf[dim].astype(float)
    ldf.rename({dim: 'val'}, axis='columns', inplace=True)
    ldf.set_index('word', inplace=True)
    return ldf


def tokenize_japanese(text, tokenizer):
    # Tokenize Japanese text using BertTokenizer
    tokens = tokenizer.tokenize(text)
    return tokens


def get_vals(twt, lexdf, tokenizer):
    tokens = tokenize_japanese(twt, tokenizer)
    pw = [x for x in tokens if x in lexdf.index]
    pv = [lexdf.loc[w]['val'] for w in pw]

    numTokens = len(tokens)
    numLexTokens = len(pw)

    if numLexTokens == 0:
        avgLexVal = 0
    else:
        avgLexVal = np.mean(pv)

    return [numTokens, numLexTokens, avgLexVal]


def process_df(df, lexdf, tokenizer):
    logging.info("Number of rows: " + str(len(df)))

    resrows = [get_vals(x, lexdf, tokenizer) for x in df['text']]
    resrows = [x + y for x, y in zip(df.values.tolist(), resrows)]

    resdf = pd.DataFrame(resrows, columns=df.columns.tolist() + ['numTokens', 'numLexTokens', 'avgLexVal'])

    # Check if any lexicon tokens were found in the text
    resdf['lexRatio'] = resdf['numLexTokens'] / resdf['numTokens']
    resdf = resdf[resdf['lexRatio'] > 0]  # Adjust threshold as needed

    logging.info("Number of rows after processing: " + str(len(resdf)))

    return resdf


def main(dataPath, lexPath, lexNames, savePath):
    os.makedirs(savePath, exist_ok=True)

    df = pd.read_csv(dataPath, encoding='utf-8')

    lexicon = read_lexicon(lexPath, lexNames)

    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    for lexName in lexNames:
        lexdf = prep_dim_lexicon(lexicon, lexName)
        logging.info(lexName + " lexicon length: " + str(len(lexdf)))
        resdf = process_df(df, lexdf, tokenizer)

        resdf.to_csv(os.path.join(savePath, lexName + '.csv'), index=False)

if __name__ == '__main__':
    args = parser.parse_args()

    main(args.dataPath, args.lexPath, args.lexNames, args.savePath)
