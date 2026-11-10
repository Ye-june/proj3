/* Chat client for Harbor. Sends questions to the Python API and renders text only. */
(function () {
  "use strict";

  const thread = document.getElementById("thread");
  const empty = document.getElementById("empty");
  const messages = document.getElementById("messages");
  const form = document.getElementById("composer");
  const input = document.getElementById("message");
  const sendButton = document.getElementById("send");
  const newChat = document.getElementById("new-chat");
  const modeStatus = document.getElementById("mode-status");

  const history = [];
  let pending = false;

  input.addEventListener("input", resize);
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const text = input.value.trim();
    if (!text || pending) {
      return;
    }
    input.value = "";
    resize();
    ask(text);
  });

  document.querySelectorAll(".prompt").forEach(function (button) {
    button.addEventListener("click", function () {
      if (pending) {
        return;
      }
      ask(button.getAttribute("data-prompt"));
    });
  });

  newChat.addEventListener("click", function () {
    if (pending) {
      return;
    }
    history.length = 0;
    messages.replaceChildren();
    empty.hidden = false;
    newChat.disabled = true;
    input.focus();
  });

  function resize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  async function ask(text) {
    empty.hidden = true;
    appendMessage("user", text);
    history.push({ role: "user", content: text });
    newChat.disabled = true;
    setPending(true);
    const prior = history.slice(0, -1);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ message: text, history: prior }),
      });
      const data = await response.json().catch(function () {
        return {};
      });
      if (!response.ok) {
        throw new Error(data.error || "Harbor could not answer just now. Please try again.");
      }
      appendMessage("assistant", data.reply || "", data.sources || []);
      history.push({ role: "assistant", content: data.reply || "" });
      if (data.mode) {
        setMode(data.mode);
      }
    } catch (error) {
      const unreachable = error instanceof TypeError;
      appendMessage(
        "error",
        unreachable
          ? "Harbor is unreachable. Check your connection and try again."
          : error.message || "Harbor could not answer just now. Please try again."
      );
    } finally {
      setPending(false);
      newChat.disabled = history.length === 0;
      input.focus();
    }
  }

  function setPending(isPending) {
    pending = isPending;
    sendButton.disabled = isPending;
    input.disabled = isPending;
    thread.setAttribute("aria-busy", isPending ? "true" : "false");
    const existing = document.getElementById("loading");
    if (existing) {
      existing.remove();
    }
    if (isPending) {
      const loading = buildMessage("assistant", "", []);
      loading.id = "loading";
      loading.querySelector(".bubble").replaceChildren(dots());
      messages.appendChild(loading);
      scrollToEnd();
    }
  }

  function dots() {
    const wrap = document.createElement("div");
    wrap.className = "dots";
    wrap.setAttribute("role", "status");
    const label = document.createElement("span");
    label.className = "sr-only";
    label.textContent = "Harbor is writing an answer";
    wrap.appendChild(label);
    for (let index = 0; index < 3; index += 1) {
      wrap.appendChild(document.createElement("span"));
    }
    return wrap;
  }

  function appendMessage(role, text, sources) {
    const row = buildMessage(role, text, sources || []);
    messages.appendChild(row);
    scrollToEnd();
  }

  function buildMessage(role, text, sources) {
    const row = document.createElement("article");
    row.className = "msg " + role;

    if (role === "assistant") {
      const avatar = document.createElement("span");
      avatar.className = "avatar";
      avatar.setAttribute("aria-hidden", "true");
      avatar.textContent = "H";
      row.appendChild(avatar);
    }

    const body = document.createElement("div");
    body.className = "msg-body";

    const who = document.createElement("p");
    who.className = "who";
    who.textContent = role === "user" ? "You" : role === "error" ? "Notice" : "Harbor";
    body.appendChild(who);

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    fillText(bubble, text);
    body.appendChild(bubble);

    appendSources(body, sources);

    row.appendChild(body);
    return row;
  }

  function fillText(bubble, text) {
    const blocks = String(text || "").trim().split(/\n{2,}/);
    blocks.forEach(function (block) {
      if (!block.trim()) {
        return;
      }
      const paragraph = document.createElement("p");
      block.split("\n").forEach(function (line, index) {
        if (index > 0) {
          paragraph.appendChild(document.createElement("br"));
        }
        paragraph.appendChild(document.createTextNode(line));
      });
      bubble.appendChild(paragraph);
    });
  }

  function appendSources(body, sources) {
    if (!sources || !sources.length) {
      return;
    }
    const wrap = document.createElement("div");
    wrap.className = "sources";
    sources.forEach(function (source) {
      if (!source) {
        return;
      }
      const line = document.createElement("p");
      line.className = "source";
      const bits = [];
      if (source.title) {
        bits.push(source.title);
      }
      if (source.retrieved_at) {
        bits.push("checked " + source.retrieved_at);
      }
      if (source.detail) {
        bits.push(source.detail);
      }
      if (!bits.length) {
        return;
      }
      line.appendChild(document.createTextNode(bits.join(" · ")));
      const href = safeUrl(source.url);
      if (href) {
        line.appendChild(document.createTextNode(" · "));
        const link = document.createElement("a");
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "Official record";
        line.appendChild(link);
      }
      wrap.appendChild(line);
    });
    if (wrap.childNodes.length) {
      body.appendChild(wrap);
    }
  }

  function safeUrl(url) {
    if (typeof url !== "string" || !url) {
      return "";
    }
    try {
      const parsed = new URL(url);
      if (parsed.protocol === "https:" || parsed.protocol === "http:") {
        return parsed.href;
      }
    } catch (error) {
      return "";
    }
    return "";
  }

  function setMode(mode) {
    modeStatus.dataset.mode = mode;
    if (mode === "rag") {
      modeStatus.textContent = "Retrieval connected";
      modeStatus.title = "Answers are requested from the connected retrieval service.";
    } else if (mode === "llm") {
      modeStatus.textContent = "Model on sources";
      modeStatus.title = "A language model rephrased retrieved text and was told not to add facts.";
    } else if (mode === "official") {
      modeStatus.textContent = "Weather Service";
      modeStatus.title = "This reply uses National Weather Service alerts checked at the time shown.";
    } else if (mode === "unavailable") {
      modeStatus.textContent = "Source missing";
      modeStatus.title = "Harbor does not have a live source for that request and did not invent one.";
    } else {
      modeStatus.textContent = "Sources required";
      modeStatus.title = "General guidance is labeled as such and is not a live incident report.";
    }
  }

  function scrollToEnd() {
    thread.scrollTop = thread.scrollHeight;
  }
})();
