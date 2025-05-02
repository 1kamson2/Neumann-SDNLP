import logging
import openai
from dset import FileManager

logger = logging.getLogger(__name__)


class GPTWrapper:
  def __init__(self):
    self.__api_path = FileManager().gpt_api_path
    self.__api_key = open(self.__api_path, "r").read().rstrip("\n")
    self.__client = openai.OpenAI(api_key=self.__api_key)

  def create_prompt(self, helper_content: str, user_content: str):
    def prompt_helper(role, prompt):
      return {"role": role, "prompt": prompt}

    assert user_content is not None
    assert helper_content is not None
    return {
      "HELPER": prompt_helper("system", helper_content),
      "USER": prompt_helper("user", user_content),
    }

  def send_prompt(self, model: str, **kwargs: dict):
    HELPER_DICT = kwargs.get("HELPER", None)
    PROMPT_DICT = kwargs.get("USER", None)
    assert HELPER_DICT is not None
    assert PROMPT_DICT is not None
    completion = self.__client.chat.completions.create(
      model=model,
      messages=[
        {"role": HELPER_DICT["role"], "content": HELPER_DICT["prompt"]},
        {"role": PROMPT_DICT["role"], "content": PROMPT_DICT["prompt"]},
      ],
    )
    return completion.choices[0].message


def gpt_client_builder():
  return GPTWrapper()


def gpt_save_to_file(result):
  try:
    with open(FileManager.gpt_out_path, "w") as _in:
      _in.write(result.content)
  except IOError as e:
    print(f"Failed while saving prompt: {e}")
    logger.error(f"Failed while saving prompt: {e}")
  finally:
    print("GPT Cleanup.")


def gpt_client_run(user_question):
  gpt_client = gpt_client_builder()
  prompts = gpt_client.create_prompt(
    "You are a helpful assistant", user_question
  )
  answer = gpt_client.send_prompt("gpt-4o-mini", **prompts)
  return answer
