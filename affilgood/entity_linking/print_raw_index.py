from whoosh import index
from whoosh.qparser import QueryParser
from whoosh.query import Term

def print_raw_index_entry(index_dir, org_name):
    """
    Print all fields and values for an organization in the Whoosh index.
    
    Parameters:
    - index_dir (str): Path to the Whoosh index directory
    - org_name (str): Name of the organization to search for
    """
    try:
        # Open the index
        ix = index.open_dir(index_dir)
        
        with ix.searcher() as searcher:
            # Print the schema first to understand field definitions
            print("INDEX SCHEMA:")
            for field_name, field_type in ix.schema.items():
                print(f"  {field_name}: {type(field_type).__name__}")
            print("\n")
            
            # Create a query for the exact organization name
            query = Term("ror_name", org_name)
            results = searcher.search(query, limit=1)
            
            if not results:
                # Try a more flexible query if exact match fails
                name_parser = QueryParser("ror_name", schema=ix.schema)
                query = name_parser.parse(org_name)
                results = searcher.search(query, limit=5)
                
            if not results:
                print(f"No organization found with name similar to: {org_name}")
                return
                
            # Print details of all matching organizations
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
    # Replace with your actual index directory
    index_dir = "whoosh_index"
    
    # Organization to search for
    org_name = "University of Stanford"
    
    print_raw_index_entry(index_dir, org_name)
