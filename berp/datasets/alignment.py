"""
General tools for aligning text annotations from two different sources with different
preprocessing standards.
"""

import re
from typing import List, Tuple

from colorama import Fore, Style
import pandas as pd


only_punct_re = re.compile(r"^[^A-zÀ-ž0-9]+$")


class Aligner:
    
    flag_types = {
        "recap": 0,  # the word was repeated one or more times in the FA
    }
    
    def __init__(self, tokens, words_df,
                 max_skip_patience: int = 20):
        self.tokens = tokens
        self.words_df = words_df
        
        # Tracks alignment between indices in FA corpus (original, prior to filtering)
        # and indices in tokens_flat. Third element indicates various metadata about
        # alignment (see `flag_types`).
        self.alignment: List[Tuple[int, int, int]] = []
        # Track current transaction for modifying alignment
        self.transaction = []
        
        self.tok_cursor = 0
        self.word_cursor = 0
        
        self.max_skip_patience = max_skip_patience
        self.skip_patience = self.max_skip_patience
    
    def process_token(self, token):
        return token.replace("Ġ", "").lower()

    @property
    def tok_el(self) -> str:
        return self.process_token(self.tokens[self.tok_cursor])
    
    @property
    def word_row(self) -> pd.Series:
        return self.words_df.iloc[self.word_cursor]
    
    @property
    def word_el(self) -> str:
        return self.word_row.word.lower()
    
    @property
    def word_index(self) -> int:
        return self.words_df.index[self.word_cursor]
    
    def advance(self, first_delta=1):
        next_token = None
        while self.tok_cursor + 1 < len(self.tokens) and \
            (next_token is None or only_punct_re.match(next_token)):
            self.tok_cursor += first_delta if next_token is None else 1

            next_token = self.tok_el

        # print("///", tok_cursor, next_token)
        
    def start_transaction(self):
        self._orig_tok_cursor = self.tok_cursor
        self._orig_word_cursor = self.word_cursor
        
    def commit_transaction(self):
        self.skip_patience = self.max_skip_patience
        self.alignment += self.transaction
        self.transaction = []
        
    def drop_transaction(self):
        self.skip_patience = self.max_skip_patience
        self.tok_cursor = self._orig_tok_cursor
        self.word_cursor = self._orig_word_cursor
        self.transaction = []
        
    def stage(self, flags=None, do_advance=True):
        print(f"{self.word_el} -- {self.tok_el}")
        self.transaction.append((self.word_index, self.tok_cursor, flags))

        # Reset skip patience
        self.skip_patience = self.max_skip_patience

        # Advance cursor
        if do_advance:
            try:
                self.advance()
            except IndexError:
                raise StopIteration
                
    def attempt_match(self) -> bool:
        fa_el = self.word_el

        if fa_el == self.tok_el:
            self.start_transaction()
            self.stage()
            self.commit_transaction()
            return True
        elif fa_el.startswith(self.tok_el):
            self.start_transaction()
            while fa_el.startswith(self.tok_el):
                fa_el = fa_el[len(self.tok_el):]
                self.stage()
                
            if len(fa_el) > 0:
                self.err(f"Residual FA el {fa_el} not covered by token {self.tok_el}. Drop transaction.")
                print(self.skip_patience)
                self.drop_transaction()
                return False
            else:
                self.commit_transaction()
                return True
        else:
            return False

    def err(self, msg):
        print(f"{Fore.RED}{msg}{Style.RESET_ALL}")
        print(self.words_df.iloc[self.word_cursor - 5 : self.word_cursor + 5])
        print(self.tokens[self.tok_cursor - 5 : self.tok_cursor + 5])
        print(self.tok_cursor)
        # raise ValueError(str((self.word_el, self.tok_el)))
        
    def __call__(self):
        # Reset state
        self.tok_cursor = 0
        self.word_cursor = 0
        self.skip_patience = self.max_skip_patience
        self.alignment = []
        
        while True:
            if self.word_cursor >= len(self.words_df):
                return self.alignment

            result = self.attempt_match()
            if result:
                self.word_cursor += 1 
            elif self.skip_patience > 0:
                # Try skipping this token and see if we find success in the near future.
                print(f"{Fore.YELLOW}Skipping token {self.tok_el}, didn't match with {self.word_el}{Style.RESET_ALL}")
                self.advance()
                self.skip_patience -= 1
            else:
                self.err("Failed to find alignment. Stop.")
                break