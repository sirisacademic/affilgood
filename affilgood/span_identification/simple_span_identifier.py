import re

def clean_whitespaces(text):
    """Clean extra whitespace from text."""
    return re.sub(r'\s+', ' ', str(text).strip())

class SimpleSpanIdentifier:
    """
    A simple implementation of span identification that treats each input text
    as a complete span without complex processing.
    
    Can optionally split text by a separator character to create multiple spans.
    """
    
    def __init__(self, separator=";", **kwargs):
        """
        Initialize the SimpleSpanIdentifier.
        
        Parameters:
        - separator (str, optional): Character to split text on. If provided,
          each substring will become a separate span. If None, the entire text
          is treated as a single span.
        - **kwargs: Additional parameters for compatibility with SpanIdentifier.
        """
        self.separator = separator
        
    def identify_spans(self, text_list):
        """
        Process a list of text data for span identification.
        
        Parameters:
        - text_list (list or str): List of strings or a single string containing text data.
        
        Returns:
        - List of dicts: Each dict contains the original text and identified spans.
        """
        # Handle single string input
        if isinstance(text_list, str):
            text_list = [text_list]
            
        # Clean each text entry
        text_list = [clean_whitespaces(text) for text in text_list]
            
        # Create results list
        results = []
        for raw_text in text_list:
            # If separator is provided, split the text by it
            if self.separator is not None:
                spans = [span.strip() for span in raw_text.split(self.separator) if span.strip()]
                # If splitting results in no valid spans, use the original text
                if not spans:
                    spans = [raw_text]
            else:
                # Otherwise, treat the whole text as a single span
                spans = [raw_text]
                
            # Add the processed data for the current text to the results
            results.append({
                "raw_text": raw_text,
                "span_entities": spans
            })
            
        return results
