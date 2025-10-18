#!/usr/bin/env python3
"""
Test script to verify the spaCy max_length fix
"""

import spacy
import sys

def test_spacy_max_length():
    """Test that spaCy can handle long texts with increased max_length"""
    
    print("Testing spaCy max_length fix...")
    
    try:
        # Load the spaCy model
        nlp = spacy.load("en_core_web_sm")
        
        # Set max_length to handle longer texts (up to 10M characters)
        nlp.max_length = 10000000
        
        print(f"✓ spaCy model loaded successfully")
        print(f"✓ max_length set to: {nlp.max_length}")
        
        # Create a long text (over 1M characters)
        long_text = "This is a test sentence. " * 300000  # About 5M characters
        print(f"✓ Created long text with {len(long_text)} characters")
        
        # Test processing the long text
        doc = nlp(long_text)
        print(f"✓ Successfully processed long text")
        print(f"✓ Processed {len(doc)} tokens")
        
        # Test with shorter text too
        short_text = "This is a short test sentence."
        doc_short = nlp(short_text)
        print(f"✓ Successfully processed short text: {len(doc_short)} tokens")
        
        return True
        
    except OSError as e:
        print(f"✗ Failed to load spaCy model: {e}")
        print("Please install the model with: python -m spacy download en_core_web_sm")
        return False
        
    except ValueError as e:
        if "exceeds maximum" in str(e):
            print(f"✗ ValueError still occurs: {e}")
            print("The fix may not be working correctly")
            return False
        else:
            print(f"✗ Unexpected ValueError: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_spacy_max_length()
    sys.exit(0 if success else 1)