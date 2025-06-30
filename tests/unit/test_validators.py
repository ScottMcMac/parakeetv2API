"""Unit tests for validators."""

import pytest

from src.core.exceptions import AudioValidationError
from src.utils.validators import (
    compare_transcriptions,
    normalize_transcription,
    sanitize_filename,
    validate_file_extension,
    validate_file_size,
)


class TestValidateFileExtension:
    """Test file extension validation."""

    def test_valid_extensions(self):
        """Test valid file extensions."""
        valid_files = [
            ("test.wav", "wav"),
            ("audio.mp3", "mp3"),
            ("speech.flac", "flac"),
            ("recording.m4a", "m4a"),
            ("video.mp4", "mp4"),
            ("stream.ogg", "ogg"),
            ("podcast.webm", "webm"),
            ("music.mpeg", "mpeg"),
            ("voice.mpga", "mpga"),
        ]
        
        for filename, expected_ext in valid_files:
            extension = validate_file_extension(filename)
            assert extension == expected_ext

    def test_invalid_extensions(self):
        """Test invalid file extensions."""
        invalid_files = [
            "test.txt",
            "audio.doc", 
            "speech.pdf",
            "data.csv",
            "archive.zip"
        ]
        
        for filename in invalid_files:
            with pytest.raises(AudioValidationError, match="Unsupported file format"):
                validate_file_extension(filename)

    def test_no_extension(self):
        """Test file without extension."""
        with pytest.raises(AudioValidationError, match="No file extension found"):
            validate_file_extension("no_extension")

    def test_case_insensitive(self):
        """Test case insensitive extension handling."""
        assert validate_file_extension("test.WAV") == "wav"
        assert validate_file_extension("audio.MP3") == "mp3"
        assert validate_file_extension("SPEECH.FLAC") == "flac"


class TestValidateFileSize:
    """Test file size validation."""

    def test_valid_sizes(self):
        """Test valid file sizes."""
        # Should not raise exception
        validate_file_size(1024)  # 1KB
        validate_file_size(1024 * 1024)  # 1MB
        validate_file_size(10 * 1024 * 1024)  # 10MB
        validate_file_size(25 * 1024 * 1024)  # 25MB (default limit)

    def test_too_large(self):
        """Test file too large."""
        with pytest.raises(AudioValidationError, match="File too large"):
            validate_file_size(30 * 1024 * 1024)  # 30MB (exceeds default 25MB limit)

    def test_zero_size(self):
        """Test zero file size."""
        # Zero size should be valid (empty file)
        validate_file_size(0)


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_basic_sanitization(self):
        """Test basic filename sanitization."""
        assert sanitize_filename("test.wav") == "test.wav"
        assert sanitize_filename("normal_file.mp3") == "normal_file.mp3"

    def test_space_replacement(self):
        """Test space replacement."""
        assert sanitize_filename("audio file.mp3") == "audio_file.mp3"
        assert sanitize_filename("my audio file.wav") == "my_audio_file.wav"

    def test_path_traversal_protection(self):
        """Test protection against path traversal."""
        assert sanitize_filename("../../../etc/passwd") == "passwd.audio"
        assert sanitize_filename("/etc/passwd") == "passwd.audio"
        assert sanitize_filename("../test.wav") == "test.wav"

    def test_special_characters(self):
        """Test special character handling."""
        result = sanitize_filename("test@#$%^&*().wav")
        assert result == "test_________.wav"

    def test_long_filename(self):
        """Test long filename truncation."""
        long_name = "a" * 300 + ".wav"
        result = sanitize_filename(long_name)
        assert len(result) <= 255
        assert result.endswith(".wav")

    def test_no_extension_handling(self):
        """Test handling of files without extension."""
        result = sanitize_filename("noextension")
        assert result == "noextension.audio"

    def test_multiple_dots(self):
        """Test handling of filenames with multiple dots."""
        assert sanitize_filename("test.file.wav") == "test.file.wav"
        assert sanitize_filename("my.audio.file.mp3") == "my.audio.file.mp3"


class TestNormalizeTranscription:
    """Test transcription normalization."""

    def test_basic_normalization(self):
        """Test basic text normalization."""
        assert normalize_transcription("Hello World!") == "hello world"
        assert normalize_transcription("THE QUICK BROWN FOX.") == "the quick brown fox"

    def test_punctuation_removal(self):
        """Test punctuation removal."""
        text = "Hello, world! How are you? I'm fine."
        expected = "hello world how are you i m fine"
        assert normalize_transcription(text) == expected

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "  Hello    world  \n\t  "
        expected = "hello world"
        assert normalize_transcription(text) == expected

    def test_numbers_preserved(self):
        """Test that numbers are preserved."""
        text = "The year is 2025 and it's great!"
        expected = "the year is 2025 and it s great"
        assert normalize_transcription(text) == expected

    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_transcription("") == ""
        assert normalize_transcription("   ") == ""

    def test_special_characters(self):
        """Test various special characters."""
        text = "This—is—a—test... with «quotes» and [brackets]!"
        expected = "this is a test with quotes and brackets"
        assert normalize_transcription(text) == expected


class TestCompareTranscriptions:
    """Test transcription comparison."""

    def test_exact_match_strict(self):
        """Test exact string match in strict mode."""
        assert compare_transcriptions("hello", "hello", strict=True)
        assert not compare_transcriptions("hello", "Hello", strict=True)
        assert not compare_transcriptions("hello", "hello.", strict=True)

    def test_exact_mismatch_strict(self):
        """Test exact string mismatch in strict mode."""
        assert not compare_transcriptions("hello", "world", strict=True)
        assert not compare_transcriptions("hello world", "hello", strict=True)

    def test_normalized_match(self):
        """Test normalized comparison."""
        text1 = "The quick brown fox jumped over the lazy dog."
        text2 = "the quick-brown fox jumped over the lazy Dog!"
        assert compare_transcriptions(text1, text2, strict=False)

        # Test expected transcription format
        expected = "The quick brown fox jumped over the lazy dog"
        variations = [
            "the quick brown fox jumped over the lazy dog.",
            "The quick-brown fox jumped over the lazy Dog!",
            "  THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG  ",
        ]
        
        for variation in variations:
            assert compare_transcriptions(expected, variation, strict=False)

    def test_normalized_mismatch(self):
        """Test normalized comparison with different content."""
        text1 = "The quick brown fox"
        text2 = "The quick brown cat"
        assert not compare_transcriptions(text1, text2, strict=False)

        # Should not match "fax" instead of "fox"
        expected = "The quick brown fox jumped over the lazy dog"
        wrong = "The quick brown fax jumped over the lazy dog"
        assert not compare_transcriptions(expected, wrong, strict=False)

    def test_default_mode(self):
        """Test default comparison mode (non-strict)."""
        text1 = "Hello World!"
        text2 = "hello world"
        assert compare_transcriptions(text1, text2)  # Default is strict=False

    def test_empty_strings(self):
        """Test comparison with empty strings."""
        assert compare_transcriptions("", "", strict=True)
        assert compare_transcriptions("", "", strict=False)
        assert not compare_transcriptions("hello", "", strict=True)
        assert not compare_transcriptions("hello", "", strict=False)