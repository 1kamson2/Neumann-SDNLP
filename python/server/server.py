import socket
import json
from typing import Callable

from server.handle_request import request_parser

HOST: str = "127.0.0.1"
PORT: int = 9999 
MAX_BUF_SIZE: int = 8192

RESP_OK_IMAGE = "HTTP/1.1 200 OK\nAccess-Control-Allow-Origin: *\nAccess-Control-Allow-Headers: Content-Type\nContent-Type: image/png\nContent-Length: {0}\n\n"
RESP_OK_TEXT = "HTTP/1.1 200 OK\nAccess-Control-Allow-Origin: *\nAccess-Control-Allow-Headers: Content-Type\nContent-Type: application/json\nContent-Length: {0}\n\n{1}"
RESP_ER = "HTTP/1.1 500 Internal Server Error\nAccess-Control-Allow-Origin: *\nAccess-Control-Allow-Headers: Content-Type\nContent-Type: application/json\nContent-Length: {0}\n\n{1}"



def client_handler(conn: socket.socket, addr: tuple[str, int]) -> None:
    """
        Handle each incoming request.

        Arguments:
            conn: Socket representing new connection.
            addr: Address of the requester.

        Returns:
            None. 
    """
    print(f"[INFO] Request from: {addr}")
    buf: bytes = conn.recv(MAX_BUF_SIZE)

    if not buf:
        print(f"[INFO] No data from: {addr}. Cleaning up.")
        return

    # buf_decoded: str = buf.decode()
    content_parsed: str = request_parser(buf)

    if len(content_parsed) == 0:
        # handshake
        body = json.dumps({"message": "Hello!"})
        sz = len(body)
        conn.sendall(RESP_OK_TEXT.format(sz, body).encode())
        conn.close()
        return

    _json: dict
    fn: Callable[[float], float]

    try:
        _json: dict = json.loads(content_parsed)
    except Exception as e:
        # json load failed
        body = json.dumps({"error": str(e)})
        sz = len(body)
        conn.sendall(RESP_ER.format(sz, body).encode())
        conn.close()
        return

    try:
        fn_str = _json.get("function")
        if not fn_str:
            body = json.dumps({"error": "Missing 'function' key in JSON"})
            sz = len(body)
            conn.sendall(RESP_ER.format(sz, body).encode())
            conn.close()
            return

        fn = get_function(fn_str)
        if isinstance(fn, Exception):
            body = json.dumps({"error": str(fn)})
            sz = len(body)
            conn.sendall(RESP_ER.format(sz, body).encode())
            conn.close()
            return

        img_bytes = render_plt_img(fn)
        img_sz = len(img_bytes)
        response = RESP_OK_IMAGE.format(img_sz).encode("utf-8") + img_bytes
        conn.sendall(response)
        conn.close()

    except Exception as e:
        # failed to process the function or create the image
        body = json.dumps({"error": str(e)})
        sz = len(body)
        conn.sendall(RESP_ER.format(sz, body).encode())
        conn.close()


def server_main() -> None:
    """
        Bind to the socket and listen for incoming requests.
    """
    print(f"[INFO] Listening for requests on:\n    {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        while True:
            conn, addr = s.accept()
            client_handler(conn, addr)
