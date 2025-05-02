const BACKEND_SERVER = "http://127.0.0.1:8080";

const Tokens = {
  NEW_CHAT: 1,
  CHAT_ACTIVE: 2,
};

class ChatManager {
  static name = "ChatManager";
  /*
   *  Implements the logic for managing chat interface.
   *
   *  Attributes:
   *    message_count: Counts the messages the user sent.
   */
  constructor() {
    this.user_message_count = 0;
  }

  async getPostResponse(userPrompt) {
    /*
     *  Requests the information from the server with user's prompt.
     *
     *  Parameters:
     *    userPrompt: The user's message.
     *
     *  Returns:
     *    Returns the whole response.
     */
    const response = await fetch(BACKEND_SERVER, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      mode: "cors",
      body: JSON.stringify(userPrompt),
    });
    return response;
  }

  async processRequest(prompt) {
    /*
     *  Process the user's message.
     *
     *  Parameters:
     *    prompt: User's message.
     *
     *  Returns:
     *    Only the body of the response.
     */
    let response = await this.getPostResponse(prompt);
    if (!response.ok) {
      console.log(`[ERROR] Fetching the response failed.\nGot: ${response}`);
    }
    return await response.text();
  }

  async processMessage(content) {
    /*
     *  Process the message, fetch the received data, create user's message,
     *  create the message with LLM's response.
     *
     *  Parameters:
     *    content: Create the message with the user's content.
     *
     * */
    console.log(
      `${ChatManager.name} is processing:\nUser prompted: ${content}`,
    );
    try {
      const dataReceived = await processRequest(content);
      console.log(`${ChatManager.name}: Model responded: ${dataReceived}.`);
      createMessage(content);
      createMessage(dataReceived);
    } catch (err) {
      console.log(err);
    }
  }

  createMessage(content) {
    /*
     * Create the message.
     */
    const newDiv = document.createElement("div");
    const newNode = document.createTextNode(content);
    const newMessaageDiv = document.getElementById("model-n-user-messages");
    newDiv.classList.add("message");
    newDiv.appendChild(newNode);
    newMessaageDiv.appendChild(newDiv);
    userInput.innerText = "";
  }
}

class NeumannSite {
  static name = "Neumann Site";
  /*
   *  Class for managing all the elements of the current chat.
   *
   *  Attributes:
   *    currentChat: A Chat Manager instance for the current chat.
   *    currentChatName: The name for the current chat.
   *    chatType: Check if the user just started the chat. (Might be used for
   *    further development)
   */

  constructor() {
    this.currentChat = new ChatManager();
    this.currentChatName = "";
    this.chatType = Tokens["NEW_CHAT"];
    userInput.addEventListener("keydown", this.chatHandler.bind(this));
  }

  chatHandler(event) {
    /*
     *  Handle key events.
     *
     *  Parameters:
     *    event: Event that occurred.
     */
    switch (event.key) {
      case "Enter": {
        event.preventDefault();
        let message = userInput.innerText;
        if (this.chatType == Tokens["NEW_CHAT"] && message.length > 0) {
          this.chatType = Tokens["CHAT_ACTIVE"];
          this.currentChatName = message;
          console.log(
            `${NeumannSite.name}: A new conversation '${this.currentChatName}'`,
          );
        }
        this.currentChat.user_message_count++;
        this.currentChat.processRequest(message);
        break;
      }
    }
  }
}

const userInput = document.getElementById("prompt");
const eHandler = new NeumannSite();
