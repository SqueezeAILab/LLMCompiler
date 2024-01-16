from src.agents.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia


def generate_tools(args):
    web_searcher = ReActWikipedia()
    if args.model_type == "vllm":
        # If we use LLaMA with vLLM for the movie recommendation task,
        # we frequently get the context length error, so we limit the
        # wikipedia context length to 400 and only return the first sentence.
        docstore = DocstoreExplorer(web_searcher, char_limit=400, one_sentence=True)
    else:
        docstore = DocstoreExplorer(web_searcher)

    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            # NOTE: This description is not used
            description="useful for when you need to ask with search",
        ),
    ]
    return tools
