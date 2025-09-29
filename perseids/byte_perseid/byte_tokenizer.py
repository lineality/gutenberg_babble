"""
ByteTokenizer: A simple byte-level tokenizer with special tokens.

Vocabulary:
    0-255: Raw bytes
    256: PAD (padding token)
    257: EOS (end of sequence)
    258: MASKUNK (mask/unknown token)
"""

import traceback
import time
from pathlib import Path


class ByteTokenizer:
    """
    ByteTokenizer: A simple byte-level tokenizer with special tokens.

    Vocabulary:
        0-255: Raw bytes
        256: PAD (padding token)
        257: EOS (end of sequence)
        258: MASKUNK (mask/unknown token)
    """

    # Special token IDs
    PAD_ID = 256
    EOS_ID = 257
    MASKUNK_ID = 258

    # Total vocabulary size
    VOCAB_SIZE = 259

    def __init__(self):
        """Initialize the ByteTokenizer."""
        # Special token string representations (for compatibility)
        self.pad_token = "<pad>"
        self.eos_token = (
            "<eos>"  # Note: eos_token (singular) - this was the original name
        )
        self.maskunk_token = "<maskunk>"

        # Map special token strings to IDs
        self.special_tokens = {
            self.pad_token: self.PAD_ID,
            self.eos_token: self.EOS_ID,
            self.maskunk_token: self.MASKUNK_ID,
        }

        # Reverse mapping for decoding
        self.special_ids = {
            self.PAD_ID: self.pad_token,
            self.EOS_ID: self.eos_token,
            self.MASKUNK_ID: self.maskunk_token,
        }

        # For tracking decode errors
        self.last_decode_errors = []

    @property
    def vocab_size(self):
        """Return the vocabulary size."""
        return self.VOCAB_SIZE

    def encode(self, text, add_eos=False, handle_special_tokens=True):
        """
        Encode text to token IDs.

        Args:
            text (str): Input text to encode
            add_eos (bool): Whether to add EOS token at the end
            handle_special_tokens (bool): Whether to parse special token strings

        Returns:
            list[int]: Token IDs (0-258)
        """
        token_ids = []

        # Handle special tokens in the text if requested
        if handle_special_tokens:
            # Simple approach: look for special token strings
            # More sophisticated: could use regex for better parsing
            remaining = text
            while remaining:
                # Check if text starts with a special token
                found_special = False
                for token_str, token_id in self.special_tokens.items():
                    if remaining.startswith(token_str):
                        token_ids.append(token_id)
                        remaining = remaining[len(token_str) :]
                        found_special = True
                        break

                if not found_special:
                    # Process one character as bytes
                    try:
                        # Convert first character to bytes
                        char = remaining[0]
                        char_bytes = char.encode("utf-8")
                        token_ids.extend(list(char_bytes))
                        remaining = remaining[1:]
                    except UnicodeEncodeError:
                        # If encoding fails, use MASKUNK
                        token_ids.append(self.MASKUNK_ID)
                        remaining = remaining[1:]
        else:
            # Simple byte encoding without special token parsing
            try:
                text_bytes = text.encode("utf-8")
                token_ids = list(text_bytes)
            except UnicodeEncodeError as e:
                # Handle encoding errors by replacing problematic characters
                text_bytes = text.encode("utf-8", errors="replace")
                token_ids = list(text_bytes)
                # Replace the Unicode replacement character bytes with MASKUNK
                # UTF-8 replacement character is 0xEF 0xBF 0xBD
                for i in range(len(token_ids) - 2):
                    if token_ids[i : i + 3] == [0xEF, 0xBF, 0xBD]:
                        token_ids[i : i + 3] = [self.MASKUNK_ID]

        # Add EOS token if requested
        if add_eos:
            token_ids.append(self.EOS_ID)

        return token_ids

    def encode_bytes(self, byte_data, add_eos=False):
        """
        Directly encode raw bytes to token IDs - no conversion needed.

        This is the most efficient method when working with file bytes,
        as bytes ARE already the tokens (0-255).

        Args:
            byte_data (bytes): Raw bytes to encode
            add_eos (bool): Whether to add EOS token at the end

        Returns:
            list[int]: Token IDs (0-258)

        Raises:
            TypeError: If byte_data is not bytes type
        """
        try:
            if not isinstance(byte_data, bytes):
                raise TypeError(
                    f"encode_bytes expects bytes, got {type(byte_data).__name__}"
                )

            # Direct conversion - bytes are already tokens!
            token_ids = list(byte_data)

            if add_eos:
                token_ids.append(self.EOS_ID)

            return token_ids

        except Exception as e:
            print(f"Error in encode_bytes: {e}")
            traceback.print_exc()
            raise

    def encode_file(self, file_path, add_eos=False, chunk_size=None, verbose=True):
        """
        Encode a file directly from its bytes - efficient for large files.

        Reads file in binary mode and converts bytes directly to tokens,
        avoiding unnecessary byte->string->byte conversions.

        Args:
            file_path (str or Path): Path to file to encode
            add_eos (bool): Whether to add EOS token at the end
            chunk_size (int or None): If provided, read file in chunks of this size
                                      for memory efficiency. None reads entire file.
            verbose (bool): Whether to print progress information

        Returns:
            list[int]: Token IDs (0-258)

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")

            file_size = file_path.stat().st_size

            if verbose:
                print(f"Encoding file: {file_path.name}")
                print(
                    f"File size: {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)"
                )
                print(f"Method: Direct byte encoding (most efficient)")
                if chunk_size:
                    print(f"Reading in chunks of {chunk_size:,} bytes")

            token_ids = []
            start_time = time.perf_counter()

            with open(file_path, "rb") as file_handle:
                if chunk_size is None:
                    # Read entire file at once
                    byte_data = file_handle.read()
                    token_ids = list(byte_data)
                else:
                    # Read in chunks for memory efficiency
                    bytes_read = 0
                    last_report_time = start_time

                    while True:
                        chunk = file_handle.read(chunk_size)
                        if not chunk:
                            break

                        # Direct conversion of bytes to tokens
                        token_ids.extend(list(chunk))
                        bytes_read += len(chunk)

                        # Progress reporting
                        if verbose and time.perf_counter() - last_report_time > 1.0:
                            progress = (bytes_read / file_size) * 100
                            elapsed = time.perf_counter() - start_time
                            speed = bytes_read / elapsed / (1024 * 1024)  # MB/s
                            print(
                                f"  Progress: {progress:.1f}% ({bytes_read:,}/{file_size:,} bytes) - {speed:.1f} MB/s"
                            )
                            last_report_time = time.perf_counter()

            if add_eos:
                token_ids.append(self.EOS_ID)

            if verbose:
                elapsed = time.perf_counter() - start_time
                speed = file_size / elapsed / (1024 * 1024)  # MB/s
                tokens_per_sec = len(token_ids) / elapsed
                print(f"✅ Tokenization complete!")
                print(f"  Time: {elapsed:.2f} seconds")
                print(f"  Speed: {speed:.2f} MB/s ({tokens_per_sec:,.0f} tokens/sec)")
                print(f"  Total tokens: {len(token_ids):,}")

            return token_ids

        except Exception as e:
            print(f"Error in encode_file: {e}")
            traceback.print_exc()
            raise

    def stream_encode_file(self, file_path, chunk_size=8192, add_eos=False):
        """
        Stream-encode a file, yielding chunks of tokens for memory efficiency.

        Generator that yields token chunks without loading entire file.
        Ideal for very large files or streaming processing.

        Args:
            file_path (str or Path): Path to file to encode
            chunk_size (int): Number of bytes to read per chunk (default 8KB)
            add_eos (bool): Whether to add EOS token to final chunk

        Yields:
            list[int]: Chunks of token IDs (0-258)

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "rb") as file_handle:
                while True:
                    chunk = file_handle.read(chunk_size)
                    if not chunk:
                        # End of file
                        if add_eos:
                            yield [self.EOS_ID]
                        break

                    # Direct conversion - bytes are tokens
                    yield list(chunk)

        except Exception as e:
            print(f"Error in stream_encode_file: {e}")
            traceback.print_exc()
            raise

    def decode(self, token_ids, skip_special_tokens=False, replace_errors=True):
        """
        Decode token IDs back to text.

        Args:
            token_ids (list[int]): Token IDs to decode
            skip_special_tokens (bool): Whether to skip special tokens in output
            replace_errors (bool): Replace invalid UTF-8 with � or raise error

        Returns:
            str: Decoded text
        """
        self.last_decode_errors = []  # Reset error tracking
        byte_list = []
        special_tokens_output = []

        for idx, token_id in enumerate(token_ids):
            if token_id < 256:
                # Regular byte
                byte_list.append(token_id)
            elif token_id in self.special_ids:
                # Special token
                if byte_list:
                    # Decode accumulated bytes before the special token
                    text_part = self._decode_bytes(byte_list, idx, replace_errors)
                    special_tokens_output.append(text_part)
                    byte_list = []

                if not skip_special_tokens:
                    special_tokens_output.append(self.special_ids[token_id])
            else:
                # Invalid token ID
                self.last_decode_errors.append((idx, token_id, "Invalid token ID"))
                if not skip_special_tokens:
                    special_tokens_output.append(f"<INVALID_{token_id}>")

        # Decode any remaining bytes
        if byte_list:
            text_part = self._decode_bytes(byte_list, len(token_ids), replace_errors)
            special_tokens_output.append(text_part)

        return "".join(special_tokens_output)

    def _decode_bytes(self, byte_list, position, replace_errors):
        """
        Helper to decode a list of bytes to text.

        Args:
            byte_list (list[int]): Bytes to decode
            position (int): Position in token stream (for error reporting)
            replace_errors (bool): How to handle UTF-8 errors

        Returns:
            str: Decoded text
        """
        try:
            byte_array = bytes(byte_list)
            if replace_errors:
                return byte_array.decode("utf-8", errors="replace")
            else:
                return byte_array.decode("utf-8")
        except UnicodeDecodeError as e:
            self.last_decode_errors.append((position, byte_list, str(e)))
            if replace_errors:
                # Use replacement character
                return "�"
            else:
                raise

    def get_stats(self, token_ids):
        """
        Get statistics about a token sequence.

        Args:
            token_ids (list[int]): Token IDs to analyze

        Returns:
            dict: Statistics about the tokens
        """
        stats = {
            "total_tokens": len(token_ids),
            "byte_tokens": sum(1 for t in token_ids if t < 256),
            "pad_tokens": sum(1 for t in token_ids if t == self.PAD_ID),
            "eos_tokens": sum(1 for t in token_ids if t == self.EOS_ID),
            "maskunk_tokens": sum(1 for t in token_ids if t == self.MASKUNK_ID),
            "invalid_tokens": sum(1 for t in token_ids if t >= self.VOCAB_SIZE),
        }
        stats["special_tokens"] = (
            stats["pad_tokens"] + stats["eos_tokens"] + stats["maskunk_tokens"]
        )
        return stats


'''
class DocumentDataset(Dataset):
    """
    Dataset for document-based training.
    Handles chunking and tokenization of text documents.

    Now supports efficient file-based tokenization to avoid
    unnecessary byte->string->byte conversions.
    """

    def __init__(self, data_source, tokenizer, max_length, stride, verbose=True, is_file=False):
        """
        Initialize document dataset.

        Args:
            data_source: Either a string of text or a file path (when is_file=True)
            tokenizer: Tokenizer object with encode method
            max_length (int): Maximum sequence length
            stride (int): Stride between chunks (for overlap)
            verbose (bool): Print statistics
            is_file (bool): If True, data_source is a file path to read directly
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        try:
            if verbose:
                print(f"\n{'='*60}")
                print("TOKENIZATION STARTING")
                print(f"{'='*60}")

            start_time = time.perf_counter()

            if is_file:
                # Use efficient file encoding - direct bytes to tokens
                file_path = Path(data_source)
                file_size = file_path.stat().st_size

                if verbose:
                    print(f"Source: File - {file_path.name}")
                    print(f"Size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
                    print(f"Method: Direct byte encoding (no string conversion)")
                    print(f"Tokenizing...")

                # Use efficient file encoding
                # For large files (>100MB), use chunked reading
                if file_size > 100 * 1024 * 1024:
                    self.tokens = tokenizer.encode_file(
                        file_path,
                        chunk_size=1024*1024,  # 1MB chunks
                        verbose=verbose
                    )
                else:
                    self.tokens = tokenizer.encode_file(file_path, verbose=verbose)

            else:
                # Regular text encoding
                if verbose:
                    print(f"Source: Text string")
                    print(f"Length: {len(data_source):,} characters")
                    print(f"Method: UTF-8 text encoding")
                    print(f"Tokenizing...")

                self.tokens = tokenizer.encode(data_source)

            elapsed = time.perf_counter() - start_time

            if verbose:
                tokens_per_sec = len(self.tokens) / elapsed if elapsed > 0 else 0
                print(f"\n✅ Tokenization complete!")
                print(f"  Time: {elapsed:.2f} seconds")
                print(f"  Speed: {tokens_per_sec:,.0f} tokens/second")
                print(f"  Total tokens: {len(self.tokens):,}")

            # Create overlapping windows
            if verbose:
                print(f"\n{'-'*60}")
                print(f"Creating training windows...")

            window_start = time.perf_counter()
            self.windows = []

            for i in range(0, len(self.tokens) - max_length, stride):
                window = self.tokens[i : i + max_length + 1]  # +1 for target
                if len(window) == max_length + 1:
                    self.windows.append(window)

            window_time = time.perf_counter() - window_start

            if verbose:
                print(f"✅ Window creation complete!")
                print(f"  Time: {window_time:.2f} seconds")
                print(f"  Training windows: {len(self.windows):,}")
                print(f"  Window size: {max_length}")
                print(f"  Stride: {stride}")
                print(f"  Overlap: {((max_length - stride) / max_length * 100):.1f}%")
                print(f"\n{'='*60}")
                print(f"DATASET READY")
                print(f"{'='*60}")

        except Exception as e:
            print(f"Error during dataset creation: {e}")
            traceback.print_exc()
            raise

    def __len__(self):
        """Return the number of training windows."""
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            tuple: (input_ids, target_ids) where target is input shifted by 1
        """
        window = self.windows[idx]
        # Input is all tokens except the last
        input_ids = torch.tensor(window[:-1], dtype=torch.long)
        # Target is all tokens except the first (shifted by 1)
        target_ids = torch.tensor(window[1:], dtype=torch.long)
        return input_ids, target_ids
'''

# Example usage
if __name__ == "__main__":
    # Run the comprehensive test suite
    # run_all_tokenizer_tests()

    # Interactive demo
    print("\n" + "=" * 60)
    print("Interactive Demo - Try your own text!")
    print("=" * 60)
    print("Type 'quit' to exit")

    tokenizer = ByteTokenizer()

    while True:
        print()
        text = input("Enter text to tokenize: ")
        if text.lower() == "quit":
            break

        tokens = tokenizer.encode(text, add_eos=True)
        decoded = tokenizer.decode(tokens)

        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Decoded: {repr(decoded)}")

        stats = tokenizer.get_stats(tokens)
        print(f"Stats: {stats['byte_tokens']} bytes, {stats['special_tokens']} special")
