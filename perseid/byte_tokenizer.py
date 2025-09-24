"""
ByteTokenizer: A simple byte-level tokenizer with special tokens.

Vocabulary:
    0-255: Raw bytes
    256: PAD (padding token)
    257: EOS (end of sequence)
    258: MASKUNK (mask/unknown token)
"""

class ByteTokenizer:
    """Byte-level tokenizer with special tokens."""
    
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
        self.eos_token = "<eos>"
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
                        remaining = remaining[len(token_str):]
                        found_special = True
                        break
                
                if not found_special:
                    # Process one character as bytes
                    try:
                        # Convert first character to bytes
                        char = remaining[0]
                        char_bytes = char.encode('utf-8')
                        token_ids.extend(list(char_bytes))
                        remaining = remaining[1:]
                    except UnicodeEncodeError:
                        # If encoding fails, use MASKUNK
                        token_ids.append(self.MASKUNK_ID)
                        remaining = remaining[1:]
        else:
            # Simple byte encoding without special token parsing
            try:
                text_bytes = text.encode('utf-8')
                token_ids = list(text_bytes)
            except UnicodeEncodeError as e:
                # Handle encoding errors by replacing problematic characters
                text_bytes = text.encode('utf-8', errors='replace')
                token_ids = list(text_bytes)
                # Replace the Unicode replacement character bytes with MASKUNK
                # UTF-8 replacement character is 0xEF 0xBF 0xBD
                for i in range(len(token_ids) - 2):
                    if token_ids[i:i+3] == [0xEF, 0xBF, 0xBD]:
                        token_ids[i:i+3] = [self.MASKUNK_ID]
        
        # Add EOS token if requested
        if add_eos:
            token_ids.append(self.EOS_ID)
        
        return token_ids
    
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
        
        return ''.join(special_tokens_output)
    
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
                return byte_array.decode('utf-8', errors='replace')
            else:
                return byte_array.decode('utf-8')
        except UnicodeDecodeError as e:
            self.last_decode_errors.append((position, byte_list, str(e)))
            if replace_errors:
                # Use replacement character
                return '�'
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
            'total_tokens': len(token_ids),
            'byte_tokens': sum(1 for t in token_ids if t < 256),
            'pad_tokens': sum(1 for t in token_ids if t == self.PAD_ID),
            'eos_tokens': sum(1 for t in token_ids if t == self.EOS_ID),
            'maskunk_tokens': sum(1 for t in token_ids if t == self.MASKUNK_ID),
            'invalid_tokens': sum(1 for t in token_ids if t >= self.VOCAB_SIZE),
        }
        stats['special_tokens'] = stats['pad_tokens'] + stats['eos_tokens'] + stats['maskunk_tokens']
        return stats


# Test the tokenizer
def test_byte_tokenizer():
    """Test suite for ByteTokenizer."""
    print("=" * 60)
    print("ByteTokenizer Test Suite")
    print("=" * 60)
    
    tokenizer = ByteTokenizer()
    
    # Test 1: Basic ASCII encoding/decoding
    print("\n1. Basic ASCII text:")
    text1 = "Hello, World!"
    tokens1 = tokenizer.encode(text1)
    decoded1 = tokenizer.decode(tokens1)
    print(f"   Original: {repr(text1)}")
    print(f"   Tokens: {tokens1}")
    print(f"   Decoded: {repr(decoded1)}")
    print(f"   Match: {text1 == decoded1}")
    
    # Test 2: UTF-8 multi-byte characters
    print("\n2. UTF-8 multi-byte characters:")
    text2 = "Hello 世界! 🌍"
    tokens2 = tokenizer.encode(text2)
    decoded2 = tokenizer.decode(tokens2)
    print(f"   Original: {repr(text2)}")
    print(f"   Tokens ({len(tokens2)}): {tokens2}")
    print(f"   Decoded: {repr(decoded2)}")
    print(f"   Match: {text2 == decoded2}")
    
    # Test 3: Special tokens
    print("\n3. Adding EOS token:")
    text3 = "End of sequence"
    tokens3 = tokenizer.encode(text3, add_eos=True)
    decoded3 = tokenizer.decode(tokens3)
    print(f"   Original: {repr(text3)}")
    print(f"   Tokens: {tokens3}")
    print(f"   Decoded: {repr(decoded3)}")
    print(f"   Has EOS: {tokenizer.EOS_ID in tokens3}")
    
    # Test 4: Special tokens in text
    print("\n4. Special tokens in text:")
    text4 = "Start<eos>Middle<pad>End"
    tokens4 = tokenizer.encode(text4, handle_special_tokens=True)
    decoded4 = tokenizer.decode(tokens4)
    decoded4_skip = tokenizer.decode(tokens4, skip_special_tokens=True)
    print(f"   Original: {repr(text4)}")
    print(f"   Tokens: {tokens4}")
    print(f"   Decoded: {repr(decoded4)}")
    print(f"   Decoded (skip special): {repr(decoded4_skip)}")
    
    # Test 5: Invalid UTF-8 handling
    print("\n5. Invalid byte sequences:")
    # Simulate corrupted bytes
    invalid_tokens = [72, 101, 108, 108, 111, 0xFF, 0xFE, 32, 87, 111, 114, 108, 100]
    decoded5 = tokenizer.decode(invalid_tokens)
    print(f"   Tokens: {invalid_tokens}")
    print(f"   Decoded: {repr(decoded5)}")
    if tokenizer.last_decode_errors:
        print(f"   Errors: {tokenizer.last_decode_errors}")
    
    # Test 6: Token statistics
    print("\n6. Token statistics:")
    mixed_tokens = tokens2 + [tokenizer.PAD_ID, tokenizer.EOS_ID]
    stats = tokenizer.get_stats(mixed_tokens)
    print(f"   Tokens: {mixed_tokens}")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test 7: Vocabulary size
    print("\n7. Vocabulary info:")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.PAD_ID})")
    print(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.EOS_ID})")
    print(f"   MASKUNK token: {tokenizer.maskunk_token} (ID: {tokenizer.MASKUNK_ID})")
    
    print("\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run the test suite
    test_byte_tokenizer()
    
    # Interactive demo
    print("\n" + "=" * 60)
    print("Interactive Demo - Try your own text!")
    print("=" * 60)
    print("Type 'quit' to exit")
    
    tokenizer = ByteTokenizer()
    
    while True:
        print()
        text = input("Enter text to tokenize: ")
        if text.lower() == 'quit':
            break
        
        tokens = tokenizer.encode(text, add_eos=True)
        decoded = tokenizer.decode(tokens)
        
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Decoded: {repr(decoded)}")
        
        stats = tokenizer.get_stats(tokens)
        print(f"Stats: {stats['byte_tokens']} bytes, {stats['special_tokens']} special")
