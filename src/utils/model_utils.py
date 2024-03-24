import os
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms import OpenAI


def get_model(
    model_type,
    model_name,
    api_key,
    vllm_port,
    stream,
    temperature=0,
):
    if model_type == "openai":
        if api_key is None:
            raise ValueError("api_key must be provided for openai model")
        llm = ChatOpenAI(
            model_name=model_name,  # type: ignore
            openai_api_key=api_key,  # type: ignore
            streaming=stream,
            temperature=temperature,
        )
    elif model_type == "azure":
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_DEPLOYMENT_NAME"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            openai_api_type="azure",
            # streaming=args.stream,
        )

    elif model_type == "vllm":
        if vllm_port is None:
            raise ValueError("vllm_port must be provided for vllm model")
        if stream:
            print(
                "WARNING: vllm does not support streaming. "
                "Setting stream=False for vllm model."
            )
        llm = OpenAI(
            openai_api_key="EMPTY",
            openai_api_base=f"http://localhost:{vllm_port}/v1",
            model_name=model_name,
            temperature=temperature,
            max_retries=1,
        )

    else:
        raise NotImplementedError(f"Unknown model type: {model_type}")

    return llm
