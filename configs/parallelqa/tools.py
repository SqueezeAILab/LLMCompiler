from src.agents.tools import Tool
from src.chains.llm_math_chain import LLMMathChain
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia

from langchain.chat_models import ChatOpenAI

_MATH_DESCRIPTION = (
    "math(problem: str, context: Optional[list[str]]) -> float:\n"
    " - Solves the provided math problem.\n"
    ' - `problem` can be either a simple math problem (e.g. "1 + 3") or a word problem (e.g. "how many apples are there if there are 3 apples and 2 apples").\n'
    " - You cannot calculate multiple expressions in one call. For instance, `math('1 + 3, 2 + 4')` does not work. "
    "If you need to calculate multiple expressions, you need to call them separately like `math('1 + 3')` and then `math('2 + 4')`\n"
    " - Minimize the number of `math` actions as much as possible. For instance, instead of calling "
    '2. math("what is the 10% of $1") and then call 3. math("$1 + $2"), '
    'you MUST call 2. math("what is the 110% of $1") instead, which will reduce the number of math actions.\n'
    # Context specific rules below
    " - You can optionally provide a list of strings as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `math` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do math on it.\n"
    " - You MUST NEVER provide `search` action's outputs as a variable in the `problem` argument. "
    "This is because `search` returns a text blob that contains the information about the entity, not a number or value. "
    "Therefore, when you need to provide an output of `search` action, you MUST provide it as a `context` argument to `math` action. "
    'For example, 1. search("Barack Obama") and then 2. math("age of $1") is NEVER allowed. '
    'Use 2. math("age of Barack Obama, context=["$1"]) instead.\n'
    " - When you ask a question about `context`, specify the units. "
    'For instance, "what is xx in height?" or "what is xx in millions?" instead of "what is xx?"\n'
)


def run_llm_math_chain_factory(llm_math_chain):
    async def run_llm_math_chain(question, context=None):
        if context is None:
            prompt = question
        else:
            if len(context) == 1:
                context_str = f"Context:\n{context[0]}"
            else:
                context_strs = []
                for i, c in enumerate(context):
                    context_strs.append(f"Context {i}:\n{c}")
                context_str = "\n\n".join(context_strs)
            prompt = (
                "Answer the Question based on the Context. When you write down a expression, it MUST ONLY consists of numbers and operators. "
                "Here are some guidelines:\n\n"
                "  - When you are asked for differences, you consider the absolute value of the difference. Difference of two numbers is always positive."
                "For instance, the difference between 1 and 2 is 1, not -1.\n"
                "  - When you are applying operations (e.g. difference, summation, ratio, etc.) between multiple values in the Context, you must unify the units of those numbers. "
                "For instance, you cannot add 1 meter to 1 foot.\n"
                "     - You must pick the values in the same units if all the values are available in the same units.\n"
                "     - If not, you must convert them to the same units before applying the operation.\n"
                "  - You MUST strictly follow the unit (e.g. meter, kilometer, million, etc.) you were asked.\n"
                "     - If the Context has the numbers in same units as the question, you can directly use them.\n"
                "     - If the Context has the numbers in different units than the question, you must convert them to the units asked in the question."
                "For example, if the question asks for the distance between two cities in kilometers, but the Context has the distance in miles, "
                "you must convert the distance to kilometers.\n"
                "  - If you are asked about a particular number in millions, billions, or any other unit, the number should be written without specifying the unit. "
                "For example, if you are asked for 100 millions, it should be written as 100, not 100 million or 100,000,000.\n"
                '  - Never introduce a variable. For instance "gazelle_max_speed * 1.4" is not allowed. Pick up a correct number from the given context.\n'
                "\n"
                f"{context_str}\n\n"
                f"Question: {question}\n\n"
            )
        response = llm_math_chain.run(prompt)
        response = response.split("Answer:")[1].strip()
        try:
            response = float(response)
            # round to 3 decimal places
            response = round(response, 3)
            response = str(response)
        except:
            pass
        return response

    return run_llm_math_chain


web_searcher = ReActWikipedia()
docstore = DocstoreExplorer(web_searcher)


def generate_tools(model_name, api_key, callbacks):
    llm_math_chain = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        temperature=0,
        callbacks=callbacks,
    )

    llm_math_chain = LLMMathChain.from_llm(llm=llm_math_chain, verbose=True)
    return [
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
        Tool(
            name="math",
            func=run_llm_math_chain_factory(llm_math_chain),
            description=_MATH_DESCRIPTION,
            stringify_rule=lambda args: f"math({args[0]})",  # drop context
        ),
    ]
