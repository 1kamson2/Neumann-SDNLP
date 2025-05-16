from typing import List
from enum import Enum


class RequestTokens(Enum):
    """
        Enum representing common HTTP request tokens as byte values (in decimal).
        Control Characters:
          SP: Space (' ')
          CR: Carriage Return ('\\r')
          LF: Line Feed ('\\n')
          COLON: Colon (':')
          SLASH: Forward Slash ('/')
          QUESTION: Question Mark ('?')
          AMPERSAND: Ampersand ('&')
          EQUALS: Equals Sign ('=')
          HASH: Hash ('#')
          COMMA: Comma (',')
          SEMICOLON: Semicolon (';')
    """
    SP = 32
    CR = 13
    LF = 10
    COLON = 58
    SLASH = 47
    QUESTION = 63
    AMPERSAND = 38
    EQUALS = 61
    HASH = 35
    COMMA = 44
    SEMICOLON = 59

def request_lexer(buf: bytes) -> List:
    """
        Lexer for requests. 

        Arguments:
            buf: Request buffer in bytes.

        Returns:
            List of bytes and request tokens.
    """
    request: List = []
    for byte_ in buf:
        match byte_:
            case 10:
                request.append(RequestTokens.LF)
                break
            case 13:
                request.append(RequestTokens.CR)
                break
            case _:
                request.append(byte_)
                break
    return request


def request_parser(buf: bytes) -> bytes: 
    """
        Scan the request and parse it.

        Arguments:
            buf: The buffer to be parsed.

        Returns:
            The content of the request.
    """
    # TODO: Easy parsing, assuming no bad agents.


    
