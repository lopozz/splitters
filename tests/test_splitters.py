from transformers import AutoTokenizer
from splitters.span_processor import SpanProcessor


class TestSpanProcesor:
    span_processor = SpanProcessor(
        AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast=False), "a"
    )

    def test_whitespace_sent_divider_by_char_long_sent(self):
        """Test that the whitespace sent divider splits the text correctly."""
        text = "This is a long text with many words that should be split properly."
        max_len = 20
        result = self.span_processor.whitespace_sent_divider_by_char(text, max_len)
        print(result)

        assert isinstance(result, list)
        assert len(result) == 4
        assert [chunck["start"] for chunck in result] == [0, 20, 36, 51]
        assert [chunck["end"] for chunck in result] == [19, 35, 50, 66]
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(
            "text" in chunk and "start" in chunk and "end" in chunk for chunk in result
        )

    def test_whitespace_sent_divider_by_char_short_sent(self):
        """Test that the whitespace return the entire sentence."""
        text = "This is a short text."
        max_len = 200
        result = self.span_processor.whitespace_sent_divider_by_char(text, max_len)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == text
        assert [chunck["start"] for chunck in result] == [0]
        assert [chunck["end"] for chunck in result] == [21]
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(
            "text" in chunk and "start" in chunk and "end" in chunk for chunk in result
        )

    def test_sent_divider_by_token_long_sent(self):
        """Test that sent_divider_by_token correctly splits tokens."""
        text = "This is a long text with many words that should be split properly."
        max_len = 5
        result = self.span_processor.sent_divider_by_token(text, max_len)

        assert isinstance(result, list)
        assert len(result) == 3
        assert [chunck["start"] for chunck in result] == [0, 5, 10]
        assert [chunck["end"] for chunck in result] == [4, 9, 13]
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(
            "text" in chunk and "start" in chunk and "end" in chunk for chunk in result
        )

    def test_sent_divider_by_token_short_sent(self):
        """Test that sent_divider_by_token correctly splits tokens."""
        text = "This is a short text."
        max_len = 200
        result = self.span_processor.sent_divider_by_token(text, max_len)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == text
        assert [chunck["start"] for chunck in result] == [0]
        assert [chunck["end"] for chunck in result] == [5]
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(
            "text" in chunk and "start" in chunk and "end" in chunk for chunk in result
        )

    def test_spacy_sent_divider(self):
        """Test that the spacy_sent_divider_it method returns the correct spans."""

        text = "This is a sentence. Here is another."
        result = self.span_processor.spacy_sent_divider(text)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["text"] == "This is a sentence."
        assert result[1]["text"] == "Here is another."

    def test_token_split_by_max_len_2_split(self):
        """Test the token_split_by_max_len method."""
        text = "This is a test sentence. Here is another sentence."
        max_len = 10
        result = self.span_processor.token_split_by_max_len(text, max_len)

        assert isinstance(result, list)
        assert len(result) == 2
        assert [chunck["start"] for chunck in result] == [0, 25]
        assert [chunck["end"] for chunck in result] == [24, 50]
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(
            "text" in chunk and "start" in chunk and "end" in chunk for chunk in result
        )

    def test_token_split_by_max_len_no_split(self):
        """Test the token_split_by_max_len method."""
        text = "This is a test sentence. Here is another sentence."
        max_len = 100
        result = self.span_processor.token_split_by_max_len(text, max_len)

        assert isinstance(result, list)
        assert len(result) == 1
        assert [chunck["start"] for chunck in result] == [0]
        assert [chunck["end"] for chunck in result] == [50]
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(
            "text" in chunk and "start" in chunk and "end" in chunk for chunk in result
        )

    def test_token_split_by_max_len_low_max_len(self):
        """Test the token_split_by_max_len method."""
        text = "This is a test sentence. Here is another sentence."
        max_len = 4
        result = self.span_processor.token_split_by_max_len(text, max_len)

        assert isinstance(result, list)
        assert len(result) == 4
        assert [chunck["start"] for chunck in result] == [0, 15, 25, 50]
        assert [chunck["end"] for chunck in result] == [13, 23, 48, 50]
        assert all(isinstance(chunk, dict) for chunk in result)
        assert all(
            "text" in chunk and "start" in chunk and "end" in chunk for chunk in result
        )
