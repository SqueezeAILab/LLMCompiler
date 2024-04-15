from src.agents.tools import Tool
from src.chains.llm_math_chain import LLMMathChain
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia
from src.utils.model_utils import get_model


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


def generate_tools(args, model_name):
    llm_math_chain = get_model(
        model_type=args.model_type,
        model_name=model_name,
        vllm_port=args.vllm_port,
        stream=False,
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
