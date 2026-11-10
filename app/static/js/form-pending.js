"use strict";

document.querySelectorAll(".chat-sidebar-delete-form").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!form.querySelector('input[name="chat_ids"]:checked')) {
      event.preventDefault();
      const pendingStatus = form.querySelector("[data-pending-status]");
      if (pendingStatus) {
        pendingStatus.textContent = "Выберите чаты для удаления.";
        pendingStatus.hidden = false;
      }
    }
  });
});

document.querySelectorAll("[data-document-chat-form]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (!form.querySelector('input[name="document_ids"]:checked')) {
      event.preventDefault();
      const pendingStatus = form.querySelector("[data-pending-status]");
      if (pendingStatus) {
        pendingStatus.textContent = "Выберите документы для чата.";
        pendingStatus.hidden = false;
      }
    }
  });
});

document.querySelectorAll("form[data-pending-message]").forEach((form) => {
  form.addEventListener("submit", (event) => {
    if (event.defaultPrevented || form.getAttribute("aria-busy") === "true") {
      event.preventDefault();
      return;
    }

    if (
      form.dataset.confirmMessage &&
      !window.confirm(form.dataset.confirmMessage)
    ) {
      event.preventDefault();
      return;
    }

    const pendingStatus = form.querySelector("[data-pending-status]");
    const submitButton = form.querySelector('button[type="submit"]');

    if (!pendingStatus || !submitButton) {
      return;
    }

    pendingStatus.textContent = form.dataset.pendingMessage;
    pendingStatus.hidden = false;
    submitButton.textContent =
      form.dataset.pendingLabel || submitButton.textContent;
    submitButton.disabled = true;
    form.setAttribute("aria-busy", "true");

    const messageInput = form.querySelector('input[name="content"]');
    const messageList = document.querySelector(".message-list");

    if (
      form.classList.contains("chat-composer") &&
      messageInput &&
      messageList
    ) {
      const content = messageInput.value.trim();
      if (!content) {
        return;
      }

      const submittedContent = document.createElement("input");
      submittedContent.type = "hidden";
      submittedContent.name = messageInput.name;
      submittedContent.value = content;
      form.append(submittedContent);
      messageInput.removeAttribute("name");

      messageList.querySelector(".empty-chat")?.remove();

      const messageCard = document.createElement("li");
      messageCard.className = "message-card message-user";

      const messageHeader = document.createElement("div");
      messageHeader.className = "message-header";

      const messageRole = document.createElement("span");
      messageRole.className = "message-role";
      messageRole.textContent = "Вы";

      const messageTime = document.createElement("span");
      messageTime.className = "message-time";
      messageTime.textContent = new Date()
        .toLocaleString("sv-SE", { hour12: false })
        .slice(0, 16);

      const messageText = document.createElement("div");
      messageText.className = "message-text";
      messageText.textContent = content;

      messageHeader.append(messageRole, messageTime);
      messageCard.append(messageHeader, messageText);
      messageList.append(messageCard);
      messageInput.value = "";
      messageList.scrollTop = messageList.scrollHeight;
    }
  });
});
