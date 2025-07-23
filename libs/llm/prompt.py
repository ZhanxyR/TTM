import importlib

from libs.utils.logger import get_logger

class PromptWrapper:
    def __init__(self, language, default_language='zh', logger=None):
        self.supported_language_list = {
            'zh': 'libs.llm.prompts.prompts_zh',
            'en': 'libs.llm.prompts.prompts_en'
        }

        self.language = language
        self.module_name = self.get_module(language)

        self.default_language = default_language
        self.default_module_name = self.get_module(default_language)

        self.logger = logger if logger is not None else get_logger()

    def _load_func(self, prompt_type, module_name):
        module = importlib.import_module(module_name)
        return getattr(module, prompt_type)
    
    def get_module(self, language):
        if language in self.supported_language_list:
            return self.supported_language_list[language]
        else:
            raise ValueError(f'Language \'{language}\' is not in the support list: {list(self.supported_language_list.keys())}.')
        
    def add_language(self, language, module_name):
        self.supported_language_list[language] = module_name

    def switch(self, language):
        self.language = language
        self.module_name = self._init_module(language)
    
    def __call__(self, prompt_type, **kwargs):
        try:
            func = self._load_func(prompt_type, self.module_name)
            result = func(**kwargs)
        except Exception as e:
            self.logger.error(f'\'{prompt_type}\' is not defined in \'{self.module_name}\'.')

            if self.language != self.default_language:
                self.logger.warning(f'Try with the default language type \'{self.default_language}\'.')
                func = self._load_func(prompt_type, self.default_module_name)
                result = func(**kwargs)
            else:
                raise e
                # self.logger.exception(e)

        return result
    




