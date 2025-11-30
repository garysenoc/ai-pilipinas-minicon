"""
Main script to run all conversational history technique examples.

This script demonstrates all 10 different techniques for implementing
conversational history in AI agents using LangChain.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your OpenAI API key.")
    sys.exit(1)

def run_technique(technique_num, technique_name, module_name, func_name):
    """Run a specific technique and handle errors gracefully."""
    print("\n" + "=" * 80)
    print(f"Running Technique {technique_num}: {technique_name}")
    print("=" * 80)
    
    try:
        # Import and run the technique
        module = __import__(f"techniques.{module_name}", fromlist=[module_name])
        
        if hasattr(module, func_name):
            getattr(module, func_name)()
        else:
            print(f"Warning: Function '{func_name}' not found in {module_name}")
    except Exception as e:
        print(f"Error running {technique_name}: {str(e)}")
        print("Skipping to next technique...\n")
        import traceback
        traceback.print_exc()

def main():
    """Run all conversational history techniques."""
    
    print("=" * 80)
    print("Conversational History Techniques with LangChain")
    print("=" * 80)
    print("\nThis script will demonstrate 11 different techniques for")
    print("implementing conversational history in AI agents.")
    print("Compatible with LangChain v1.0+\n")
    
    # List of all techniques (num, name, module, function)
    techniques = [
        (1, "Basic Buffer Memory", "01_basic_buffer_memory", "demonstrate_buffer_memory"),
        (2, "Summary Memory", "02_summary_memory", "demonstrate_summary_memory"),
        (3, "Window Memory", "03_window_memory", "demonstrate_window_memory"),
        (4, "Summary Buffer Memory", "04_summary_buffer_memory", "demonstrate_summary_buffer_memory"),
        (5, "Entity Memory", "05_entity_memory", "demonstrate_entity_memory"),
        (6, "Knowledge Graph Memory", "06_knowledge_graph_memory", "demonstrate_kg_memory"),
        (7, "Vector Store Memory", "07_vector_store_memory", "demonstrate_vector_store_memory"),
        (8, "Redis-Backed Memory", "08_redis_memory", "demonstrate_redis_memory"),
        (9, "SQL-Backed Memory", "09_sql_memory", "demonstrate_sql_memory"),
        (10, "Custom Memory", "10_custom_memory", "demonstrate_custom_memory"),
        (11, "LCEL with Message History (v1.0+)", "11_lcel_memory", "demonstrate_lcel_memory"),
    ]
    
    # Ask user which techniques to run
    print("Options:")
    print("1. Run all techniques")
    print("2. Run specific technique(s)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "3":
        print("Exiting...")
        return
    elif choice == "2":
        print("\nAvailable techniques:")
        for num, name, _ in techniques:
            print(f"  {num}. {name}")
        
        selected = input("\nEnter technique numbers (comma-separated, e.g., 1,2,3): ").strip()
        try:
            selected_nums = [int(x.strip()) for x in selected.split(",")]
            techniques_to_run = [t for t in techniques if t[0] in selected_nums]
        except ValueError:
            print("Invalid input. Running all techniques...")
            techniques_to_run = techniques
    else:
        techniques_to_run = techniques
    
    # Run selected techniques
    for num, name, module, func in techniques_to_run:
        run_technique(num, name, module, func)
        input("\nPress Enter to continue to next technique...")
    
    print("\n" + "=" * 80)
    print("All demonstrations complete!")
    print("=" * 80)
    print("\nFor detailed explanations, see README.md")
    print("For individual technique code, see techniques/ directory")
    print("\nNote: Technique 11 (LCEL) is the recommended approach for")
    print("new LangChain v1.0+ projects. See langchain.com for latest docs.")

if __name__ == "__main__":
    main()

