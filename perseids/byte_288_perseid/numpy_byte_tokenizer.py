"""
byte_tokenizer.py

A simple byte-level tokenizer with special tokens for deep learning models.
Converts text to raw bytes (0-255) plus special tokens for PAD, EOS, and MASKUNK.
"""

import numpy as np
from typing import List, Optional, Union
from pathlib import Path
import json


class ByteTokenizer:
    """
    Byte-level tokenizer that encodes text as raw bytes with special tokens.
    
    Vocabulary:
        0-255: Raw byte values
        256: PAD (padding token)
        257: EOS (end of sequence)
        258: MASKUNK (mask/unknown token for corrupted data)
    """
    
    # Special token IDs
    PAD_ID = 256
    EOS_ID = 257
    MASKUNK_ID = 258
    
    # Special token strings (for compatibility with existing interfaces)
    PAD_TOKEN = "<pad>"
    EOS_TOKEN = "<eos>"
    MASKUNK_TOKEN = "<maskunk>"
    
    def __init__(self, add_eos: bool = True, handle_errors: str = "replace"):
        """
        Initialize the ByteTokenizer.
        
        Args:
            add_eos: Whether to automatically append EOS token when encoding
            handle_errors: How to handle UTF-8 decode errors:
                - "replace": Replace bad bytes with MASKUNK token
                - "ignore": Skip bad bytes
                - "strict": Raise an error
        """
        self.add_eos = add_eos
        self.handle_errors = handle_errors
        
        # For interface compatibility
        self.eos_token = self.EOS_TOKEN
        self.pad_token = self.PAD_TOKEN
        self.unk_token = self.MASKUNK_TOKEN
        
        # Token ID mappings
        self.special_tokens = {
            self.PAD_TOKEN: self.PAD_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.MASKUNK_TOKEN: self.MASKUNK_ID,
        }
        
        self.special_ids_to_tokens = {v: k for k, v in self.special_tokens.items()}
    
    @property
    def vocab_size(self) -> int:
        """Return the total vocabulary size."""
        return 259  # 256 bytes + 3 special tokens
    
    def encode(self, text: str, add_eos: Optional[bool] = None) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text to encode
            add_eos: Override default add_eos behavior
            
        Returns:
            List of token IDs (0-258)
        """
        if add_eos is None:
            add_eos = self.add_eos
        
        # Convert text to bytes
        try:
            byte_values = text.encode('utf-8')
            token_ids = list(byte_values)  # Convert bytes to list of ints (0-255)
        except UnicodeEncodeError as e:
            # Should rarely happen with valid Python strings
            print(f"Warning: Encoding error at position {e.start}-{e.end}")
            # Fallback: encode what we can
            byte_values = text.encode('utf-8', errors='replace')
            token_ids = list(byte_values)
        
        # Add EOS token if requested
        if add_eos:
            token_ids.append(self.EOS_ID)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back into text.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        output_bytes = []
        special_token_strs = []
        
        for token_id in token_ids:
            if token_id < 256:
                # Regular byte value
                output_bytes.append(token_id)
            elif token_id in self.special_ids_to_tokens:
                # Special token
                if not skip_special_tokens:
                    # Add special token string representation if we have accumulated bytes
                    if output_bytes:
                        # First decode accumulated bytes
                        text_part = self._decode_bytes(output_bytes)
                        special_token_strs.append(text_part)
                        output_bytes = []
                    # Add special token representation
                    special_token_strs.append(self.special_ids_to_tokens[token_id])
            else:
                # Invalid token ID - treat as MASKUNK
                print(f"Warning: Invalid token ID {token_id}, treating as MASKUNK")
                if not skip_special_tokens:
                    if output_bytes:
                        text_part = self._decode_bytes(output_bytes)
                        special_token_strs.append(text_part)
                        output_bytes = []
                    special_token_strs.append(self.MASKUNK_TOKEN)
        
        # Decode any remaining bytes
        if output_bytes:
            text_part = self._decode_bytes(output_bytes)
            special_token_strs.append(text_part)
        
        return ''.join(special_token_strs)
    
    def _decode_bytes(self, byte_list: List[int]) -> str:
        """
        Helper to decode a list of byte values to string.
        
        Args:
            byte_list: List of integers (0-255)
            
        Returns:
            Decoded string
        """
        try:
            byte_array = bytes(byte_list)
            if self.handle_errors == "strict":
                return byte_array.decode('utf-8')
            elif self.handle_errors == "ignore":
                return byte_array.decode('utf-8', errors='ignore')
            else:  # "replace" mode
                # Use the Unicode replacement character for bad sequences
                return byte_array.decode('utf-8', errors='replace')
        except Exception as e:
            print(f"Error decoding bytes: {e}")
            if self.handle_errors == "strict":
                raise
            # Return replacement characters for the whole sequence
            return '�' * len(byte_list)
    
    def encode_batch(self, texts: List[str], add_eos: Optional[bool] = None) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text, add_eos=add_eos) for text in texts]
    
    def decode_batch(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode multiple token ID sequences."""
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in token_ids_batch]
    
    def pad_sequences(self, sequences: List[List[int]], max_length: Optional[int] = None, 
                     padding_side: str = "right") -> np.ndarray:
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of token ID sequences
            max_length: Maximum length (if None, uses longest sequence)
            padding_side: "right" or "left"
            
        Returns:
            Numpy array of padded sequences
        """
        if not sequences:
            return np.array([[]], dtype=np.int32)
        
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded = np.full((len(sequences), max_length), self.PAD_ID, dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), max_length)
            if padding_side == "right":
                padded[i, :seq_len] = seq[:seq_len]
            else:  # left padding
                padded[i, -seq_len:] = seq[:seq_len]
        
        return padded
    
    def save(self, path: Union[str, Path]):
        """Save tokenizer configuration."""
        config = {
            "tokenizer_type": "ByteTokenizer",
            "vocab_size": self.vocab_size,
            "add_eos": self.add_eos,
            "handle_errors": self.handle_errors,
            "special_tokens": self.special_tokens,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ByteTokenizer':
        """Load tokenizer from configuration."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(
            add_eos=config.get("add_eos", True),
            handle_errors=config.get("handle_errors", "replace")
        )


# Test functions
def test_byte_tokenizer():
    """Comprehensive test suite for ByteTokenizer."""
    
    print("="*60)
    print("Testing ByteTokenizer")
    print("="*60)
    
    # Initialize tokenizer
    tokenizer = ByteTokenizer(add_eos=True)
    
    # Test 1: Basic ASCII encoding/decoding
    print("\n1. Basic ASCII Test:")
    text = "Hello, World!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"   Original: '{text}'")
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: '{decoded}'")
    assert decoded == text, "ASCII round-trip failed"
    print("   ✓ Passed")
    
    # Test 2: UTF-8 multi-byte characters
    print("\n2. UTF-8 Multi-byte Test:")
    text = "Hello 世界! 🌍"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"   Original: '{text}'")
    print(f"   Token count: {len(tokens)}")
    print(f"   First 20 tokens: {tokens[:20]}")
    print(f"   Decoded: '{decoded}'")
    assert decoded == text, "UTF-8 round-trip failed"
    print("   ✓ Passed")
    
    # Test 3: Special tokens
    print("\n3. Special Tokens Test:")
    tokens_with_eos = tokenizer.encode("Test", add_eos=True)
    tokens_without_eos = tokenizer.encode("Test", add_eos=False)
    print(f"   With EOS: {tokens_with_eos}")
    print(f"   Without EOS: {tokens_without_eos}")
    assert tokens_with_eos[-1] == ByteTokenizer.EOS_ID, "EOS not added"
    assert tokens_without_eos[-1] != ByteTokenizer.EOS_ID, "EOS incorrectly added"
    print("   ✓ Passed")
    
    # Test 4: Decode with special tokens visible
    print("\n4. Special Tokens Visibility Test:")
    manual_tokens = [72, 101, 108, 108, 111, ByteTokenizer.PAD_ID, ByteTokenizer.EOS_ID]
    decoded_skip = tokenizer.decode(manual_tokens, skip_special_tokens=True)
    decoded_show = tokenizer.decode(manual_tokens, skip_special_tokens=False)
    print(f"   Tokens: {manual_tokens}")
    print(f"   Decoded (skip special): '{decoded_skip}'")
    print(f"   Decoded (show special): '{decoded_show}'")
    assert decoded_skip == "Hello", "Special tokens not skipped"
    assert "<pad>" in decoded_show and "<eos>" in decoded_show, "Special tokens not shown"
    print("   ✓ Passed")
    
    # Test 5: Padding sequences
    print("\n5. Sequence Padding Test:")
    sequences = [
        tokenizer.encode("Short"),
        tokenizer.encode("A longer sequence here"),
        tokenizer.encode("Mid")
    ]
    padded = tokenizer.pad_sequences(sequences)
    print(f"   Original lengths: {[len(s) for s in sequences]}")
    print(f"   Padded shape: {padded.shape}")
    print(f"   First sequence padded: {padded[0].tolist()}")
    assert padded.shape[0] == 3, "Wrong batch size"
    assert all(padded[i, -1] == ByteTokenizer.PAD_ID for i in [0, 2]), "Padding not applied"
    print("   ✓ Passed")
    
    # Test 6: Batch operations
    print("\n6. Batch Operations Test:")
    texts = ["First", "Second", "Third"]
    batch_encoded = tokenizer.encode_batch(texts)
    batch_decoded = tokenizer.decode_batch(batch_encoded)
    print(f"   Original: {texts}")
    print(f"   Decoded: {batch_decoded}")
    assert batch_decoded == texts, "Batch round-trip failed"
    print("   ✓ Passed")
    
    # Test 7: Invalid token handling
    print("\n7. Invalid Token Handling Test:")
    invalid_tokens = [72, 105, 300, 400, 33, ByteTokenizer.EOS_ID]  # 300, 400 are invalid
    decoded = tokenizer.decode(invalid_tokens, skip_special_tokens=False)
    print(f"   Tokens (with invalid): {invalid_tokens}")
    print(f"   Decoded: '{decoded}'")
    print("   ✓ Handled gracefully")
    
    # Test 8: Corrupted UTF-8 handling
    print("\n8. Corrupted UTF-8 Test:")
    # Create invalid UTF-8 sequence
    invalid_utf8_tokens = [0xFF, 0xFE, 0xFD]  # Invalid UTF-8 bytes
    decoded = tokenizer.decode(invalid_utf8_tokens)
    print(f"   Invalid byte sequence: {invalid_utf8_tokens}")
    print(f"   Decoded (with replacement): '{decoded}'")
    print("   ✓ Handled gracefully")
    
    # Test 9: Vocabulary size
    print("\n9. Vocabulary Size Test:")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size == 259, "Wrong vocabulary size"
    print("   ✓ Correct size (256 bytes + 3 special)")
    
    # Test 10: Empty input
    print("\n10. Edge Cases Test:")
    empty_encoded = tokenizer.encode("", add_eos=False)
    empty_decoded = tokenizer.decode([])
    print(f"   Empty string encoded: {empty_encoded}")
    print(f"   Empty list decoded: '{empty_decoded}'")
    assert empty_encoded == [], "Empty string encoding failed"
    assert empty_decoded == "", "Empty list decoding failed"
    print("   ✓ Passed")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    # Run the test suite
    test_byte_tokenizer()
    
    print("\n" + "="*60)
    print("Interactive Demo")
    print("="*60)
    
    # Interactive demo
    tokenizer = ByteTokenizer()
    
    while True:
        text = input("\nEnter text to tokenize (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"\nOriginal text: '{text}'")
        print(f"Token IDs: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Decoded text: '{decoded}'")
        print(f"Round-trip success: {text == decoded}")
