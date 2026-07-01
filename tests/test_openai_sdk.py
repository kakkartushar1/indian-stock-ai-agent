import unittest

import openai_sdk


class OpenAISDKWrapperTests(unittest.TestCase):
    def test_sdk_imports_from_site_packages_when_venv_is_inside_repo(self):
        self.assertTrue(openai_sdk.SDK_AVAILABLE, openai_sdk.get_import_error())
        self.assertTrue(hasattr(openai_sdk, "OpenAIChatCompletionsModel"))
        self.assertTrue(hasattr(openai_sdk, "AsyncOpenAI"))


if __name__ == "__main__":
    unittest.main()
