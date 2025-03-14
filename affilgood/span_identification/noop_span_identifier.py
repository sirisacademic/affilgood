import re

def clean_whitespaces(text):
    """Clean extra whitespace from text."""
    return re.sub(r'\s+', ' ', str(text).strip())

class NoopSpanIdentifier:
    """
    A span identifier that doesn't modify the input text.
    Each input text is treated as a single span without any splitting or modification.
    Useful for pre-segmented data where each input is already a single span.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the NoopSpanIdentifier.
        
        Parameters:
        - **kwargs: Additional parameters for compatibility with other SpanIdentifier classes.
        """
        pass
        
    def identify_spans(self, text_list):
        """
        Process a list of text data, treating each as a single span.
        
        Parameters:
        - text_list (list or str): List of strings or a single string containing text data.
        
        Returns:
        - List of dicts: Each dict contains the original text and itself as a single span.
        """
        # Handle single string input
        if isinstance(text_list, str):
            text_list = [text_list]
            
        # Clean each text entry
        text_list = [clean_whitespaces(text) for text in text_list]
            
        # Create results list - each text becomes exactly one span
        results = []
        for raw_text in text_list:
            results.append({
                "raw_text": raw_text,
                "span_entities": [raw_text]
            })
            
        return results

