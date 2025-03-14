import re
import warnings
import os
import sys
import io
import logging
from transformers import pipeline, logging as transformers_logging
from .base_reranker import BaseReranker

TASK_DESCRIPTION_TEXT = """
You are assistant helping to disambiguate research organization names.

You are provided with affiliation strings obtained from research publications and a list of candidate organizations.
Candidate organizations can include the organization name and location and are identified by their ROR IDs.
Your task is to evaluate these inputs and determine the most likely match between each affiliation string and a candidate organization.
When assessing the likeliness, consider that affiliation strings and candidates might be in different languages and there might be errors or omissions in the affiliation strings and/or the candidate organizations' names or locations.
When in doubt, select the most specific institution.

The output should be a tuple with the format: (most_likely_organization, most_likely_ror)
If none of the candidates is a likely match the returned tuple should be ("None", "None")

VERY IMPORTANT: Please output only the tuple WITHOUT ANY EXPLANATION.

* Example *

Affiliation: Whitehead Institute for Biomedical Research, MIT, MA
Candidates:
- Massachusetts Institute of Technology, Cambridge, United States (ROR: 01h3p4f56)
- Harvard University, Cambridge, MA, USA (ROR: 02t5q7k89)
- Whitehead Institute, Boston, US (ROR: 03z8s9r07)

Expected response:

("Whitehead Institute, Boston, US", "03z8s9r07")

"""

DEFAULT_MODEL = "TheBloke/neural-chat-7B-v3-2-GPTQ"
MAX_NEW_TOKENS = 500

class LLMReranker(BaseReranker):
    """Re-ranks entity linking predictions using an LLM model."""
            
    def __init__(self, model_name=None, verbose=False):
        self.model_name = model_name if model_name else DEFAULT_MODEL
        self.verbose = verbose
        
        # Store original logging levels
        original_tf_verbosity = transformers_logging.get_verbosity()
        original_logging_level = logging.getLogger().level
        
        try:
            if not verbose:
                # Disable all transformers logging
                transformers_logging.set_verbosity_error()
                # Suppress other logging
                logging.getLogger().setLevel(logging.ERROR)
                # Disable warnings
                warnings.filterwarnings("ignore")
                # Redirect stdout/stderr during model loading
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = io.StringIO()
                sys.stderr = io.StringIO()
                try:
                    self.pipeline = pipeline('text-generation', model=self.model_name, device_map="auto")
                finally:
                    # Restore stdout/stderr
                    sys.stdout, sys.stderr = old_stdout, old_stderr
            else:
                self.pipeline = pipeline('text-generation', model=self.model_name, device_map="auto")
        finally:
            # Restore original logging levels
            transformers_logging.set_verbosity(original_tf_verbosity)
            logging.getLogger().setLevel(original_logging_level)

    def rerank(self, affiliation, candidates):
        """
        Takes an affiliation string and a list of candidate organizations, 
        and returns the best match using LLM scoring.
        """
        prompt = self._format_prompt(affiliation, candidates)
        pad_token_id=self.pipeline.tokenizer.eos_token_id
        outputs = self.pipeline(prompt, max_new_tokens=len(prompt)+MAX_NEW_TOKENS, temperature=0.1, do_sample=True, pad_token_id=pad_token_id)
        response = outputs[0]['generated_text'].replace(prompt, '')
        return self._parse_response(response)

    def _format_prompt(self, affiliation, candidates):
        """Format prompt for LLM input."""
        organizations = f"Affiliation: {affiliation}\nCandidates:\n"
        for candidate in candidates:
            organizations += f"- {candidate}\n"
        prompt = f'<|system|>{TASK_DESCRIPTION_TEXT}\n<|user|>{organizations}\n<|assistant|>'
        return prompt

    def _parse_response(self, response):
        """Extracts the best match from the LLM output."""
        # Try to find the standard tuple format
        match = re.search(r'\(\s*"(.+?)"\s*,\s*"(.+?)"\s*\)', response)
        if match:
            ror_id = match.group(2)
            # Check if the "None" string was returned
            if ror_id.lower() == "none":
                return None
            return ror_id
        
        # Try alternative formats if standard format fails
        # Look for ROR ID pattern directly (typically in format: 0xxxxx)
        ror_match = re.search(r'(?<!\w)([0-9a-z]{8,})(?!\w)', response)
        if ror_match:
            return ror_match.group(1)
        
        # Look for text that might indicate a ROR ID
        if "ROR:" in response:
            ror_colon_match = re.search(r'ROR:\s*([0-9a-z]{8,})', response)
            if ror_colon_match:
                return ror_colon_match.group(1)
        
        # If no ROR ID can be found, return None
        return None

