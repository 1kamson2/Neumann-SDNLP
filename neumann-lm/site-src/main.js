const Tokens = {
  NEW_CHAT: 1,
  CHAT_ACTIVE: 2,
};

class EventHandler {
  static className = "EventHandler()";
  static maxTokenLength = 24;
  constructor() {
    this.chatName = "";
    this.isNewChat = Tokens["NEW_CHAT"];
    this.mesCnt = 0;
    this.listener = null;
    this.listenerInit();
  }

  listenerInit() {
    switch (this.isNewChat) {
      case Tokens["NEW_CHAT"]:
        this.listener = (event) => this.initializeChat(event);
        userInput.addEventListener("keydown", this.listener);
        break;
      case Tokens["CHAT_ACTIVE"]:
        this.listener = (event) => this.processMessage(event);
        userInput.addEventListener("keydown", this.listener);
        break;
    }
  }

  initializeChat(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      this.isNewChat = Tokens["CHAT_ACTIVE"];
      let text = userInput.innerText;

      if (text.length >= EventHandler.maxTokenLength) {
        this.chatName = text.slice(0, EventHandler.maxTokenLength - 3) + "...";
      } else {
        this.chatName = text;
      }

      this.historyEntry(this.chatName);
      this.mesCnt++;
      console.log(
        EventHandler.className +
        ":" +
        this.mesCnt +
        ": User started a new conversation:\n" +
        this.chatName,
      );
      userInput.removeEventListener("keydown", this.listener);
      this.listenerInit();
      this.processMessage(event);
    }
  }

  processMessage(event) {
    if (event.key === "Enter") {
      event.preventDefault();
      console.log(
        EventHandler.className +
        ": User typed the following prompt:\n" +
        userInput.innerText,
      );
      const mdiv = document.createElement("div");
      mdiv.classList.add("container");
      const content = document.createTextNode(userInput.innerText);
      mdiv.appendChild(content);
      const lmdiv = document.getElementById("message-container");
      lmdiv.appendChild(mdiv);
      userInput.innerText = "";
    }
  }

  historyEntry(text) {
    const hdiv = document.createElement("div");
    hdiv.classList.add("history-log");
    const content = document.createTextNode(text);
    hdiv.appendChild(content);
    const lhdiv = document.getElementById("chat-history");
    lhdiv.appendChild(hdiv);
  }
}

const userInput = document.getElementById("prompt");
const eHandler = new EventHandler();
