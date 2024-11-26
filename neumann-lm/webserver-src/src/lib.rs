mod server;
use server::app::Server;
use std::thread;
use tokio::runtime::Runtime;
const IP: &str = "127.0.0.1";
const PORT: &str = "8000";

// main function of the program
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    thread::spawn(move || {
        let mut server = Server::new(IP, PORT).expect("something");
        server.run();
    });
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
