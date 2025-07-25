#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/11/04 15:35
@Author  : Yingli 
@File    : llm_provider_registry.py
@Ref     : Based on the MetaGpt 
"""

from graphrag.Config.LLMConfig import LLMConfig, LLMType
from graphrag.Core.Provider.BaseLLM import BaseLLM


class LLMProviderRegistry:
    def __init__(self):
        self.providers = {}

    def register(self, key, provider_cls):
        self.providers[key] = provider_cls

    def get_provider(self, enum: LLMType):
        """get provider instance according to the enum"""
        return self.providers[enum]


def register_provider(keys):
    """register provider to registry"""
    def decorator(cls):
        if isinstance(keys, list):
            for key in keys:
                LLM_REGISTRY.register(key, cls)
        else:
            LLM_REGISTRY.register(keys, cls)
        return cls

    return decorator


def create_llm_instance(config: LLMConfig) -> BaseLLM:
    """get the default llm provider"""
    llm = LLM_REGISTRY.get_provider(config.api_type)(config)
    if llm.use_system_prompt and not config.use_system_prompt:
        # for models like o1-series, default openai provider.use_system_prompt is True, but it should be False for o1-*
        llm.use_system_prompt = config.use_system_prompt
    return llm


# Registry instance
LLM_REGISTRY = LLMProviderRegistry()