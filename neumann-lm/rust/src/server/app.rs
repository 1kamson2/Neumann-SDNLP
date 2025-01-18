use crate::prompt_utils::read_prompts::FileManager;
use std::net::SocketAddr;
use std::process::Command;
use std::str;
use std::{thread, time};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, Error};
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

    #[tokio::main]
    pub async fn run(&mut self) {
        /*
         * The main function of the server, that handles incoming connetions
         * and sends them to the correct functions. Supports multithreading.
         */
        let listener = TcpListener::bind(&self.addr).await.unwrap();
        println!(
            "Server host should be running now. Address: {0}.\n\n",
            self.addr
        );
        loop {
            let (stream, _addr) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                handle_connection(stream, _addr).await;
            });
        }
    }
}

async fn connection_info(buffer: &Vec<&str>, mut _addr: &SocketAddr) {
    /*
     * Function used to display basic information about incoming connection,
     * and data that comes with it
     */
    println!("Incoming connection from {}", _addr);
    println!("========== THE BEGINNING OF THE FRAME ==========");
    println!("==== CONNECTION FROM: {}", _addr);
    buffer.iter().for_each(|line| println!("==== {}", line));
    println!("========== THE END OF THE FRAME ==========");
}

async fn handle_connection(mut stream: TcpStream, mut _addr: SocketAddr) {
    /*
     * Handling all connections that have been redirected here.
     * This function can:
     *  > Answer the POST request. This will send the data returned
     *   by the models that have been implemented.
     *  > Will be able answer the GET request. This is made for the web
     *   browser users.
     */
    let mut streamb = BufReader::new(&mut stream);
    let mut buffer: [u8; 1024] = [0; 1024];
    let mut manager = FileManager::new();
    let _ = streamb.read(&mut buffer[..]).await;
    let buffer_s: Vec<&str> = str::from_utf8(&buffer).unwrap().lines().collect();
    let status = buffer_s[0];
    let _user_prompt = buffer_s[15].to_string();

    // https://users.rust-lang.org/t/how-would-you-handle-a-incoming-post-request-in-a-http-server/82590/4
    connection_info(&buffer_s, &_addr).await;
    let (status, response) = if status.trim() == "POST / HTTP/1.1" {
        Command::new(&manager.shell_catalog)
            .arg("--prompt")
            .arg(format!("\"{}\"", manager.prompt_sanitizer(_user_prompt)))
            .arg("--model")
            .arg("nlp")
            .spawn()
            .unwrap();
        thread::sleep(time::Duration::from_secs(5));
        let prompt_feedback: String = manager.read_prompt();
        let prompt_success_status: String = String::from(PROMPT_SUCCESS_STATUS);
        (prompt_success_status, prompt_feedback)
    } else {
        let prompt_failure_status = String::from(PROMPT_FAILURE_STATUS);
        let prompt_failure_content = String::from(PROMPT_FAILURE_CONTENT);
        (prompt_failure_status, prompt_failure_content)
    };
    handle_response(&mut stream, status, response).await;
}

async fn handle_response(stream: &mut TcpStream, status: String, response: String) {
    /* Function that answers user's request.*/
    let content = response;
    let sz = content.len();
    let response = format!(
        "{status}\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {sz}\r\n\r\n{content}"
    );
    println!("Sending the following response:\n{}", response);
    stream.write_all(response.as_bytes()).await.unwrap();
}
