from src.agents.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia

web_searcher = ReActWikipedia()
docstore = DocstoreExplorer(web_searcher)

tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description=("useful for when you need to ask with search"),
    ),
]
