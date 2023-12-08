from enum import Enum

from src.react.base import (
    ReActDocstoreAgentForCustom,
    ReActDocstoreAgentForMovie,
    ReActDocstoreAgentForWiki,
)


class AgentType(str, Enum):
    """Enumerator with the Agent types."""

    REACT_DOCSTORE_FOR_WIKI = "react-docstore-wiki"
    REACT_DOCSTORE_FOR_MOVIE = "react-docstore-movie"
    REACT_DOCSTORE_FOR_CUSTOM = "react-docstore-custom"


AGENT_TO_CLASS = {
    AgentType.REACT_DOCSTORE_FOR_WIKI: ReActDocstoreAgentForWiki,
    AgentType.REACT_DOCSTORE_FOR_MOVIE: ReActDocstoreAgentForMovie,
    AgentType.REACT_DOCSTORE_FOR_CUSTOM: ReActDocstoreAgentForCustom,
}
