import os
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.llms import OpenAI


def get_model(
    model_type,
    model_name,
    vllm_port,
    stream,
    temperature=0,
):
    if model_type == "openai":
        llm = ChatOpenAI(
            model_name=model_name,  # type: ignore
            openai_api_key=os.environ["OPENAI_API_KEY"],  # type: ignore
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
    elif model_type == "friendli":
        from langchain_community.llms.friendli import Friendli

        if stream:
            print(
                "WARNING: Friendli does not support streaming. "
                "Setting stream=False for friendli endpoints."
            )
        assert "FRIENDLI_TOKEN" in os.environ, "FRIENDLI_TOKEN must be provided"
        llm = Friendli(
            model=model_name,
            temperature=temperature,
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
