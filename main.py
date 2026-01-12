import os
from data_processor import process_data
from vector_engine import VectorEngine
from config import GEMINI_MODEL


def main():
    """Main program for the system"""

    print("\n" + "=" * 60)
    print("Welcome to BookMind - Smart Book Recommendation System")
    print("=" * 60 + "\n")

    print("Loading book data...")
    try:
        df = process_data()
        print(f"Successfully loaded {len(df)} books!\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("=" * 60)
    api_key = input("Enter Gemini API key: ").strip()
    print("=" * 60 + "\n")

    if not api_key:
        print("API key must be provided!")
        return

    print("Initializing search engine...")
    try:
        engine = VectorEngine(gemini_api_key=api_key)

        if engine.is_index_ready():
            print("Loading saved index...")
            engine.load_saved_index()
        else:
            print("Creating search index (this may take a minute)...")
            engine.create_embeddings(df['text'].tolist())
            engine.build_index()

        print("Engine is ready!\n")
    except Exception as e:
        print(f"Error setting up engine: {e}")
        return

    print("=" * 60)
    print("Start searching for books (type 'exit' to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("What book are you looking for? ").strip()

            if query.lower() in ['exit', 'quit']:
                print("\nThank you for using BookMind! Goodbye!")
                break

            if not query:
                print("Please enter a valid question.\n")
                continue

            print("\n" + "=" * 60)
            print("Searching and analyzing...")
            print("=" * 60 + "\n")

            engine.ask_gemini(query=query, df=df)

        except KeyboardInterrupt:
            print("\n\nProgram stopped.")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()