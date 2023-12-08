"""Chain that interprets a prompt and executes python code to do math."""
from __future__ import annotations

import ast
import math
import re
import warnings
from typing import Any, Dict, List, Optional

import numexpr
from src.chains.chain import Chain
from src.chains.llm_chain import LLMChain

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

# flake8: noqa
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel

_PROMPT_TEMPLATE = """Translate a math problem into a expression that can be executed using Python's numexpr library. Use the output of running this code to answer the question.
You MUST follow the following guidelines:
 - Do not use "where(...)" expressions in your code since it is not supported.
 - Do not use "fmax(...)" expression in your code since it is not supported. Use "max(...)" instead.
 - Never introduce a variable. For instance "gazelle_max_speed * 1.4" is not allowed. Pick up a correct number from the given context.

Question: ${{Question with math problem.}}
```text
${{single line mathematical expression that solves the problem}}
```
...numexpr.evaluate(text)...
```output
${{Output of running the code}}
```
Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?
```text
37593 * 67
```
...numexpr.evaluate("37593 * 67")...
```output
2518731
```
Answer: 2518731

Question: 37593^(1/5)
```text
37593**(1/5)
```
...numexpr.evaluate("37593**(1/5)")...
```output
8.222831614237718
```
Answer: 8.222831614237718

Question: {question}
"""

PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


# helper functions to handle min and max functions
def compute_function(match):
    func, values = match.groups()

    # Extract numbers and remove commas from between digits
    numbers = [float(re.sub(r"(?<=\d),(?=\d)", "", v)) for v in values.split(",")]

    # Compute the min or max based on the detected function
    result = min(numbers) if func == "min" else max(numbers)

    return str(result)


class MaxTransformer(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)  # Apply the transformation to child nodes first

        if isinstance(node.func, ast.Name) and node.func.id in ("max", "min"):
            if all(isinstance(arg, (ast.Num, ast.Constant)) for arg in node.args):
                # Calculate the max value
                # print(node.args)
                args_as_strings = (ast.unparse(arg) for arg in node.args)
                args_str = ", ".join(args_as_strings)
                print(args_str)
                if node.func.id == "min":
                    value = min(
                        arg.n if isinstance(arg, ast.Num) else arg.value
                        for arg in node.args
                    )
                else:
                    value = max(
                        arg.n if isinstance(arg, ast.Num) else arg.value
                        for arg in node.args
                    )
                # Replace the max call with the max value directly
                return ast.copy_location(ast.Constant(value=value), node)
        return node


def replace_min_max_functions(expression):
    # Parse the expression into an AST
    parsed_expression = ast.parse(expression, mode="eval")

    # Transform the AST
    transformer = MaxTransformer()
    transformed_ast = transformer.visit(parsed_expression)

    # Fix the missing locations in the AST
    transformed_ast = ast.fix_missing_locations(transformed_ast)

    # Compile the transformed AST
    compiled_code = compile(transformed_ast, "<string>", "eval")

    # Evaluate the compiled code
    result = eval(compiled_code)
    return str(result)


class LLMMathChain(Chain):
    """Chain that interprets a prompt and executes python code to do math.

    Example:
        .. code-block:: python

            from langchain import LLMMathChain, OpenAI
            llm_math = LLMMathChain.from_llm(OpenAI())
    """

    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated] Prompt to use to translate to python if necessary."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMMathChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _evaluate_expression(self, expression: str) -> str:
        try:
            expression = expression.strip()
            # Remove any commas in between digits from the expression
            # e.g. "1,000,000, 2,000,000" -> "1000000, 2000000"
            expression = re.sub(r"(?<=\d),(?=\d)", "", expression)
            local_dict = {"pi": math.pi, "e": math.e}

            # Handle min and max functions
            expression = replace_min_max_functions(expression)
            output = str(
                numexpr.evaluate(
                    expression,
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                )
            )
        except Exception as e:
            print(
                f'LLMMathChain._evaluate("{expression}") raised error: {e}.'
                " Please try again with a valid numerical expression"
            )
            return "error at math chain"

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)

    def _process_llm_result(
        self, llm_output: str, run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:
        run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            return {self.output_key: "Answer: error at math chain"}
            # raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    async def _aprocess_llm_result(
        self,
        llm_output: str,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])
        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return self._process_llm_result(llm_output, _run_manager)

    async def _acall(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        llm_output = await self.llm_chain.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_math_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ) -> LLMMathChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)
