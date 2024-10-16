import os
import sys
import pytest
from transformers import AutoTokenizer

from splitters.span_processor import SpanProcessor


# Sample input data for testing
SAMPLE_TEXT = "This is a test sentence. Here is another sentence."
TOKENIZED_TEXT = ["This", "is", "a", "test", "sentence", ".", "Here", "is", "another", "sentence", "."]
TOKEN_IDS = list(range(len(TOKENIZED_TEXT)))  # Mock token IDs

# @pytest.fixture
# def mock_tokenizer():
#     """Fixture to create a mock tokenizer."""
#     tokenizer = MagicMock()
#     tokenizer.tokenize.return_value = TOKENIZED_TEXT
#     tokenizer.convert_tokens_to_ids.return_value = TOKEN_IDS
#     tokenizer.convert_ids_to_tokens.return_value = TOKENIZED_TEXT
#     tokenizer.convert_tokens_to_string.return_value = SAMPLE_TEXT
#     tokenizer.unk_token = None
#     tokenizer.pad_token = None
#     return tokenizer

# @pytest.fixture
# def span_processor(mock_tokenizer):
#     """Fixture to create the SpanProcessor object with a mock tokenizer."""
#     return SpanProcessor(tokenizer=mock_tokenizer)

# class TestSpanProcesor:
#     data_processor = SpanProcessor(AutoTokenizer.from_pretrained('microsoft/deberta-v3-small'), 'a')

#     def test_whitespace_sent_divider_by_char(self):
#         """Test that the whitespace sent divider splits the text correctly."""
#         text = "This is a long text with many words that should be split properly."
#         max_len = 20
#         result = self.data_processor.whitespace_sent_divider_by_char(text, max_len)

#         assert isinstance(result, list)
#         assert len(result) > 0
#         assert all(isinstance(chunk, dict) for chunk in result)
#         assert all("text" in chunk and "start" in chunk and "end" in chunk for chunk in result)

    # def test_sent_divider_by_token(self, span_processor, mock_tokenizer):
    #     """Test that sent_divider_by_token correctly splits tokens."""
    #     text = SAMPLE_TEXT
    #     max_len = 5
    #     result = span_processor.sent_divider_by_token(text, max_len)

    #     assert isinstance(result, list)
    #     assert len(result) > 0
    #     assert all(isinstance(chunk, dict) for chunk in result)
    #     assert all("text" in chunk and "start" in chunk and "end" in chunk for chunk in result)

    # def test_spacy_sent_divider_it(self, mocker, span_processor):
    #     """Test that the spacy_sent_divider_it method returns the correct spans."""
    #     mock_nlp = mocker.patch("spacy.load")
    #     mock_doc = MagicMock()
        
    #     # Mock the sents for SpaCy
    #     mock_sent_1 = MagicMock()
    #     mock_sent_1.text = "This is a sentence."
    #     mock_sent_1.start_char = 0
    #     mock_sent_1.end_char = 18

    #     mock_sent_2 = MagicMock()
    #     mock_sent_2.text = "Here is another."
    #     mock_sent_2.start_char = 19
    #     mock_sent_2.end_char = 33

    #     mock_doc.sents = [mock_sent_1, mock_sent_2]
    #     mock_nlp.return_value = mock_doc

    #     text = "This is a sentence. Here is another."
    #     result = span_processor.spacy_sent_divider_it(text)

    #     assert isinstance(result, list)
    #     assert len(result) == 2
    #     assert result[0]["text"] == "This is a sentence."
    #     assert result[1]["text"] == "Here is another."

    # def test_token_split_by_max_len(self, span_processor, mock_tokenizer):
    #     """Test the token_split_by_max_len method."""
    #     text = SAMPLE_TEXT
    #     max_len = 5
    #     result = span_processor.token_split_by_max_len(text, max_len)

    #     assert isinstance(result, list)
    #     assert len(result) > 0
    #     assert all(isinstance(chunk, dict) for chunk in result)
    #     assert all("text" in chunk and "start" in chunk and "end" in chunk for chunk in result)
    #     # Check that no chunk exceeds max_len tokens
    #     for chunk in result:
    #         token_count = len(mock_tokenizer.tokenize(chunk["text"]))
    #         assert token_count <= max_len
