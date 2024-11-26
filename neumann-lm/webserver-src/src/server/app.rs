use std::{collections::HashMap, fs};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Error};
use tokio::net::{TcpListener, TcpStream};

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
    // what is the request?
    let mut request = BufReader::new(&mut stream);
    // read the http request
    let mut buffer: String = String::new();
    request.read_line(&mut buffer).await;
    let (status, filename) = if buffer == "GET / HTTP/1.1" {
        ("HTTP/1.1 200 OK", "../site-src/index.html")
    } else {
        ("HTTP/1.1 404 NOT FOUND", "../site-src/index.html")
    };
    handle_response(&mut stream, status, filename).await;
}

async fn handle_response(stream: &mut TcpStream, status: &str, filename: &str) {
    let content = fs::read_to_string(filename).unwrap();
    let sz = content.len();
    let response = format!("{status}\r\nContent-Length: {sz}\r\n\r\n{content}");
    stream.write_all(response.as_bytes()).await.unwrap();
}
