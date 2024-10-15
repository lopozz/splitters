import spacy

from transformers import AutoTokenizer
from typing import List, Dict


class SpanProcessor:
    # def __init__(self, config, tokenizer, words_splitter, labels_tokenizer = None, preprocess_text=False):
    def __init__(self, tokenizer=None, sent_splitter=None, max_char_len=100):
        # self.config = config
        self.tokenizer = tokenizer
        self.sent_splitter = sent_splitter
        self.max_char_len = max_char_len
        # self.labels_tokenizer = labels_tokenizer

        # self.words_splitter = words_splitter
        # self.ent_token = config.ent_token
        # self.sep_token = config.sep_token

        # self.preprocess_text = preprocess_text

        # Check if the tokenizer has unk_token and pad_token
        self._check_and_set_special_tokens(self.tokenizer)
        # if self.labels_tokenizer:
        #     self._check_and_set_special_tokens(self.labels_tokenizer)

    def _check_and_set_special_tokens(self, tokenizer):
        """
        Checks if the given tokenizer has special tokens such as `unk_token` and `pad_token`.
        If these tokens are not present, it sets default values for them and raises a warning.

        Args:
            tokenizer (Tokenizer): The tokenizer to check for special tokens.

        Raises:
            UserWarning: If the tokenizer is missing unk_token or pad_token, a warning is raised and 
                         the default values '[UNK]' and '[PAD]' are set, respectively.
        """
        # Check for unk_token
        if tokenizer.unk_token is None:
            default_unk_token = '[UNK]'
            warnings.warn(
                f"The tokenizer is missing an 'unk_token'. Setting default '{default_unk_token}'.",
                UserWarning
            )
            tokenizer.unk_token = default_unk_token

        # Check for pad_token
        if tokenizer.pad_token is None:
            default_pad_token = '[PAD]'
            warnings.warn(
                f"The tokenizer is missing a 'pad_token'. Setting default '{default_pad_token}'.",
                UserWarning
            )
            tokenizer.pad_token = default_pad_token

    def whitespace_sent_divider_by_char(self, text: str, max_char_len: int = 100) -> List[Dict[str, int|str]]:
        """
        Divide a string in smaller chunks of max size given by `max_char_len`.
        The `text` is split in the closer white space to the position 
        indicated by `max_char_len`.
        """
        chunks = []
        start = 0

        while len(text) > max_char_len:
            # Find the closest whitespace to the max_char_len
            break_index = text.rfind(' ', 0, max_char_len)
            # If no whitespace is found, break at max_char_len
            if break_index == -1:
                break_index = max_char_len
            end = start + break_index
            chunks.append({'text': text[:break_index].strip(), 'start': start, 'end': end})
            text = text[break_index:].strip()
            start = end + 1  # Account for the whitespace

        # Add the remaining text as the last chunk
        if text:
            end = start + len(text)
            chunks.append({'text': text.strip(), 'start': start, 'end': end})

        return chunks

    def sent_divider_by_token(self, text: str, max_tk_len: int = 100) -> List[Dict[str, int|str]]:
        """
        Divide a string into smaller chunks based on tokens, with each chunk containing
        a maximum number of tokens defined by `max_tk_len`.
        
        Args:
            text (str): The input text to be tokenized and split into chunks.
            max_tk_len (int): The maximum number of tokens per chunk.
        
        Returns:
            List[Dict[str, int|str]]: A list of dictionaries containing the chunked text and 
                                      corresponding start and end token positions.
        """
        # Tokenize the input text
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        chunks = []
        start_token_idx = 0

        while len(token_ids) > max_tk_len:
            # Create the chunk by selecting the first `max_tk_len` tokens
            chunk_tokens = tokens[:max_tk_len]
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            end_token_idx = start_token_idx + max_tk_len - 1

            # Append chunk info to the result list
            chunks.append({'text': chunk_text.strip(), 'start': start_token_idx, 'end': end_token_idx})

            # Remove the processed tokens from the list
            tokens = tokens[max_tk_len:]
            token_ids = token_ids[max_tk_len:]
            start_token_idx = end_token_idx + 1

        # Add the remaining tokens as the last chunk
        if tokens:
            chunk_text = self.tokenizer.convert_tokens_to_string(tokens)
            end_token_idx = start_token_idx + len(tokens) - 1
            chunks.append({'text': chunk_text.strip(), 'start': start_token_idx, 'end': end_token_idx})

        return chunks

    def spacy_sent_divider_it(self, text: str) -> Dict[str, int|str]:

        nlp = spacy.load("it_core_news_sm")
        doc = nlp(text)
        spans = []
        for span in doc.sents:
            spans.append({"text": span.text, "start": span.start_char, "end": span.end_char})

        return spans

    def whitespace_sent_divider_by_char(self):
        raise NotImplementedError("TODO")

    def token_split_by_max_len(self, text: str, max_len: int = 100) -> List[Dict[str, int|str]]:
        """
        Split a text into chunks based on a maximum token length. If a sentence exceeds the
        max_len, it will be split into smaller chunks at token-level boundaries.

        Args:
            text (str): The input text to be split.
            max_len (int): The maximum number of tokens allowed in each chunk.

        Returns:
            List[Dict[str, int|str]]: A list of dictionaries, where each dictionary 
                                    contains the chunked text and its character-level 
                                    start and end boundaries.
        """
        # Step 1: Get sentence spans using SpaCy
        spans = self.spacy_sent_divider_it(text)

        # Step 2: Initialize variables to store the final chunks
        chunks = []
        current_chunk_tokens = []
        current_chunk_text = ""
        start_char = 0

        for span in spans:
            # Tokenize the current sentence text
            sentence_tokens = self.tokenizer.tokenize(span["text"])
            token_ids = self.tokenizer.convert_tokens_to_ids(sentence_tokens)

            # Case 1: Sentence is longer than max_len, split it
            if len(token_ids) > max_len:
                token_start_char = span["start"]

                # Process the long sentence by splitting it into smaller chunks
                for i in range(0, len(token_ids), max_len):
                    chunk_tokens = token_ids[i:i+max_len]
                    chunk_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(chunk_tokens))
                    token_end_char = token_start_char + len(chunk_text)
                    
                    # Append the split chunk
                    chunks.append({
                        "text": chunk_text.strip(),
                        "start": token_start_char,
                        "end": token_end_char - 1
                    })
                    token_start_char = token_end_char + 1  # Update the start_char for the next chunk

                continue  # Continue to the next sentence

            # Case 2: Adding this sentence exceeds max_len, finalize current chunk
            if len(current_chunk_tokens) + len(token_ids) > max_len:
                chunk_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(current_chunk_tokens))
                chunks.append({
                    "text": chunk_text.strip(),
                    "start": start_char,
                    "end": span["start"] - 1  # Use the start of the new sentence as the end of the chunk
                })

                # Reset for the new chunk
                current_chunk_tokens = []
                current_chunk_text = ""
                start_char = span["start"]  # Set the start to the beginning of the next sentence

            # Case 3: Sentence can be added to the current chunk
            current_chunk_tokens.extend(token_ids)
            current_chunk_text += " " + span["text"]

        # Add the last chunk if any tokens are remaining
        if current_chunk_tokens:
            chunk_text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(current_chunk_tokens))
            chunks.append({
                "text": chunk_text.strip(),
                "start": start_char,
                "end": spans[-1]["end"]
            })

        return chunks


text = 'In 1980, Robert Plutchik diagrammed a wheel of eight emotions: joy, trust, fear, surprise, sadness, disgust, anger and anticipation, inspired by his Ten Postulates.[49][50] Plutchik also theorized twenty-four "Primary", "Secondary", and "Tertiary" dyads (feelings composed of two emotions).[51][52][53][54][55][56][57] The wheel emotions can be paired in four groups:'

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-small')  # I would like to add a 

data_processor = SpanProcessor(tokenizer, 'a')

# spans = data_processor.token_sent_divider('In 1980, Robert Plutchik diagrammed a wheel of eight emotions: joy, trust, fear, surprise, sadness, disgust, anger and anticipation, inspired by his Ten Postulates.[49][50] Plutchik also theorized twenty-four "Primary", "Secondary", and "Tertiary" dyads (feelings composed of two emotions).[51][52][53][54][55][56][57] The wheel emotions can be paired in four groups:')

spans = data_processor.spacy_sent_divider_it(text)

print(spans)

spans = data_processor.token_split_by_max_len(text)
print(spans)

