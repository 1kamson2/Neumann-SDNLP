const Tokens = {
  NEW_CHAT: 1,
  CHAT_ACTIVE: 2,
};

class ChatManager {
  static name = "ChatManager";
  constructor() {
    this.nmess = 0;
  }
  async POSTRequest(prompt) {
    try {
      const response = await fetch("//127.0.0.1:8000", {
        method: "POST",
        headers: {
          "Content-Type": "text/plain",
        },
        body: prompt.toString(),
      });

      if (!response.ok) {
        console.log(`Fetching response failed: ${response}`);
        throw new Error(`HTTP ERROR STATUS: ${response.status}`);
      }
      return await response.text();
    } catch (err) {
      console.error("Error sending message:", err);
      throw err;
    }
  }

  async processMessage(content) {
    console.log(`${ChatManager.name} USER: Prompt: ${content}.`);
    try {
      const dataRecv = await this.POSTRequest(content);
      console.log(`${ChatManager.name} LLM: Prompt: ${dataRecv}.`);
      this.createMessage(content);
      this.createMessage(dataRecv);
    } catch (err) {
      console.log(err);
    }
  }

  createMessage(content) {
    const mdiv = document.createElement("div");
    const nodeContent = document.createTextNode(content);
    const lmdiv = document.getElementById("message-container");
    mdiv.classList.add("container");
    mdiv.appendChild(nodeContent);
    lmdiv.appendChild(mdiv);
    userInput.innerText = "";
  }

  historyEntry(content) {
    const hdiv = document.createElement("div");
    const nodeContent = document.createTextNode(content);
    const lhdiv = document.getElementById("chat-history");
    hdiv.classList.add("history-log");
    hdiv.appendChild(nodeContent);
    lhdiv.appendChild(hdiv);
  }
}

class EventHandler {
  static name = "EventHandler";
  static maxTokenLength = 32;

  constructor() {
    /*
     *  Warning: For now no parameters, because there is no handling an already
     *  existing chat.
     *  Arguments:
     *    > name - Handling the name of the class.
     *    > maxTokenLength - Handling the max length, before it will be
     *      truncated in the history section.
     *    > chatName - The name of the current session, visible in the history
     *      section.
     *    > isNewChat - The flag for managing the messages.
     *    > nmess - The number of messages.
     *    > chat - This is instance of Chat Manager that will be handling
     *    outcoming / incoming messages.
     *    > listener - Handling message events, working with the Chat Manager.
     */
    this.chat = new ChatManager();
    this.chatName = "";
    this.chatType = Tokens["NEW_CHAT"];
    userInput.addEventListener("keydown", this.chatEvents.bind(this));
  }

  chatEvents(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      let text = userInput.innerText;
      if (this.chatType == Tokens["NEW_CHAT"] && text.length > 0) {
        this.chatType = Tokens["CHAT_ACTIVE"];
        if (text.length >= EventHandler.maxTokenLength) {
          this.chatName =
            text.slice(0, EventHandler.maxTokenLength - 3) + "...";
        } else {
          this.chatName = text;
        }
        this.chat.historyEntry(this.chatName);
        this.chat.nmess++;
        console.log(
          `${EventHandler.name}: A new conversation '${this.chatName}'`,
        );
        this.chat.processMessage(text);
      } else if (this.chatType == Tokens["CHAT_ACTIVE"]) {
        this.chat.processMessage(text);
      }
    }
  }
}
const userInput = document.getElementById("prompt");
const eHandler = new EventHandler();
