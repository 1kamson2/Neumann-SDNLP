use std::fs;
static GLOBAL_PROMPT_PATH: &str = "../ml-src/GPT_OUT";
pub struct Prompt {
    prompt: String,
    prompt_path: String,
}

impl Prompt {
    pub fn new() -> Self {
        Prompt {
            prompt: String::from(""),
            prompt_path: String::from(GLOBAL_PROMPT_PATH),
        }
    }

    pub fn read_prompt(&mut self) -> String {
        let content = fs::read_to_string(&self.prompt_path).unwrap();
        self.prompt = content.clone();
        return content;
    }
}
