import sys
from whoosh import index
from whoosh.query import Term

INDEX = 'whoosh_index'

def query_ror_id(ror_id):
    """
    Query a Whoosh index for a specific ROR ID and print the results.
    
    Parameters:
    - index_dir (str): Path to the Whoosh index directory
    - ror_id (str): The ROR ID to search for (can include or exclude https://ror.org/)
    """
    # Remove the URL prefix if present
    if not ror_id.startswith("https://ror.org/"):
        ror_id = f"https://ror.org/{ror_id}"
    
    try:
        # Open the index
        ix = index.open_dir(INDEX)
        
        with ix.searcher() as searcher:
            # Create a query for the exact ROR ID
            query = Term("ror_id", ror_id)
            results = searcher.search(query, limit=5)
            
            if not results:
                print(f"No organization found with ROR ID: {ror_id}")
                return
            
            print(f"Found {len(results)} matching organizations:")
            for i, org in enumerate(results):
                print(f"\nORGANIZATION #{i+1}:")
                for field_name, field_value in org.items():
                    print(f"  {field_name}: {repr(field_value)}")
                
                # Also show the stored document (which may have additional data)
                print("\n  STORED DOCUMENT:")
                stored_doc = searcher.stored_fields(org.docnum)
                for field_name, field_value in stored_doc.items():
                    print(f"    {field_name}: {repr(field_value)}")
                    
    except Exception as e:
        print(f"Error querying index: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_ror.py <ror_id>")
        sys.exit(1)
        
    ror_id = sys.argv[1]
    
    query_ror_id(ror_id)
    
