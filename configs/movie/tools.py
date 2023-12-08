from src.agents.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia

web_searcher = ReActWikipedia()
docstore = DocstoreExplorer(web_searcher)

tools = [
    Tool(
        name="search",
        func=docstore.asearch,
        description=(
            "search(entity: str) -> str:\n"
            " - Executes an exact search for the entity on Wikipedia.\n"
            " - Returns the first paragraph if the entity is found.\n"
        ),
        stringify_rule=lambda args: f"search({args[0]})",
    ),
]
