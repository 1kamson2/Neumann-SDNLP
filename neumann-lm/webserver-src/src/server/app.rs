use crate::prompt_utils::read_prompts::Prompt;
use std::process::Command;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Error};
use tokio::net::{TcpListener, TcpStream};
static PROMPT_SUCCESS_STATUS: &str = "HTTP/1.1 200";
static PROMPT_FAILURE_STATUS: &str = "HTTP/1.1 404 NOT FOUND";
static PROMPT_FAILURE_CONTENT: &str = "We were unable to read your prompt.";

#[derive(Debug)]
pub struct Server {
    ip: String,
    port: String,
    addr: String,
}

impl Server {
    // this macro makes it act like synchronous function
    #[tokio::main]
    pub async fn new(ip: &str, port: &str) -> Result<Self, Error> {
        let addr = format!("{ip}:{port}");
        println!("Starting up your server... Please wait!");
        Ok(Server {
            ip: String::from(ip),
            port: String::from(port),
            addr,
        })
    }

    // this macro makes it act like synchronous function, but still allows to call async functions.
    #[tokio::main]
    pub async fn run(&mut self) {
        let listener = TcpListener::bind(&self.addr).await.unwrap();
        println!(
            "Server host should be running now. Address: {0}.",
            self.addr
        );
        loop {
            let (stream, _addr) = listener.accept().await.unwrap();
            println!("Connection established with {0}", _addr);

            tokio::spawn(async move {
                handle_connection(stream).await;
            });
        }
    }
}
async fn handle_connection(mut stream: TcpStream) {
    let prompt_success_status = String::from(PROMPT_SUCCESS_STATUS);
    let prompt_failure_status = String::from(PROMPT_FAILURE_STATUS);
    let prompt_failure_content = String::from(PROMPT_FAILURE_CONTENT);
    // what is the request?
    let mut streamb = BufReader::new(&mut stream);
    // read the http request
    let mut buffer: String = String::new();
    let _ = streamb.read_line(&mut buffer).await;

    println!("HTTP REQUEST: {}", buffer);
    // Here we should be executing our command to produce output in GPT_OUT.
    // Use Command::new(...) or bind python's functions

    let (status, response) = if buffer.trim() == "POST / HTTP/1.1" {
        let mut prompt_interface = Prompt::new();
        let prompt_feedback = prompt_interface.read_prompt();
        (prompt_success_status, prompt_feedback)
    } else {
        (prompt_failure_status, prompt_failure_content)
    };
    // streamb.read_line(&mut buffer).await; // read next line, it appends to
    // buffer which is string, do nice way to handle the text.
    handle_response(&mut stream, status, response).await;
}

async fn handle_response(stream: &mut TcpStream, status: String, response: String) {
    //let content = fs::read_to_string(response).unwrap();
    let content = response;
    let sz = content.len();
    let response = format!(
        "{status}\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {sz}\r\n\r\n{content}"
    );
    println!("Sending the following reponse:\n{}", response);
    stream.write_all(response.as_bytes()).await.unwrap();
}
