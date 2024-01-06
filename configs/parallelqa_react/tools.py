from langchain.chat_models import ChatOpenAI

from src.agents.tools import Tool
from src.chains.llm_math_chain import LLMMathChain
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia


def run_llm_math_chain_factory(llm_math_chain):
    def run_llm_math_chain(args):
        # since llm math chain returns the answer with a prefix, we need to remove it
        try:
            response = llm_math_chain.run(args)
            try:
                r = response.split("Answer:")[-1]
                r = r.strip()
                r = float(r)
                # round to 3 decimal places
                r = round(r, 3)
                response = "Answer: " + str(r)
            except:
                pass
        except:
            response = "Error: Invalid expression. Try with a different expression."
        return response

    return run_llm_math_chain


web_searcher = ReActWikipedia()
docstore = DocstoreExplorer(web_searcher)


def generate_tools(model_name, api_key):
    llm_math_chain = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        temperature=0,
    )

    llm_math_chain = LLMMathChain.from_llm(llm=llm_math_chain, verbose=True)

    return [
        Tool(
            name="Search",
            func=docstore.search,
            description="",  # not used
        ),
        Tool(
            name="Calculate",
            func=run_llm_math_chain_factory(llm_math_chain),
            description="",  # not used
        ),
    ]
