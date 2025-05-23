use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct FileManager {
    pub last_prompt: String,
    pub prompt_path: PathBuf,
    pub root_project_path: PathBuf,
    pub rust_catalog: PathBuf,
    pub python_catalog: PathBuf,
    pub site_catalog: PathBuf,
    pub shell_catalog: PathBuf,
}

impl FileManager {
    /* TODO: ROOT_PROJECT_PATH should be parsed from "config" file or sth */
    pub fn new() -> Self {
        let root_project_path: PathBuf =
            Path::new("/home/kums0n-desktop/Dev/lm-project/neumann-lm/").to_path_buf();
        let rust_catalog: PathBuf = root_project_path.join(Path::new("rust/src/"));
        let python_catalog: PathBuf = root_project_path.join(Path::new("python/"));
        let site_catalog: PathBuf = root_project_path.join(Path::new("frontend/"));
        let prompt_path: PathBuf = python_catalog.join(Path::new("GPT_OUT"));
        let shell_catalog: PathBuf = rust_catalog.join(Path::new("model_run.sh"));
        FileManager {
            last_prompt: String::new(),
            prompt_path,
            root_project_path,
            rust_catalog,
            python_catalog,
            site_catalog,
            shell_catalog,
        }
    }

    pub fn read_prompt(&mut self) -> String {
        /*
         * Function that reads the file, then it callbacks to the function,
         * that cleans the prompt of characters, that might hinder the
         * function of the prompt parser.
         */
        let prompt: String = fs::read_to_string(&self.prompt_path).unwrap();
        return self.prompt_sanitizer(prompt);
    }

    pub fn prompt_sanitizer(&mut self, prompt: String) -> String {
        /*
         * Functions that reads prompts (for now given in the specific file)
         * and parses the output. The read file is also added to the last read
         * prompt.
         */
        let dont_keep = |c: &char| -> bool { vec!['\0', '\n', '\r'].contains(c) };
        let prompt_sanitized: String = prompt
            .chars()
            .map(|c| if dont_keep(&c) { ' ' } else { c })
            .collect::<String>()
            .trim()
            .to_string();
        self.last_prompt = prompt_sanitized.clone();
        return prompt_sanitized;
    }
}
