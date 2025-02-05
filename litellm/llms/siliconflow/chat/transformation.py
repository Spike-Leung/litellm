"""
Translates from OpenAI's `/v1/chat/completions` to Siliconflow's `/v1/chat/completions`
"""

from typing import List, Optional, Tuple

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class SiliconFlowChatConfig(OpenAIGPTConfig):

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str
    ) -> List[AllMessageValues]:
        """
        Siliconflow does not support content in list format.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        return super()._transform_messages(messages=messages, model=model)

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("SILICONFLOW_API_BASE")
            or "https://api.siliconflow.cn/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("SILICONFLOW_API_KEY")
        return api_base, dynamic_api_key
