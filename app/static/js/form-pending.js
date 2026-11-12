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

document.querySelectorAll(".chat-main").forEach((chatMain) => {
  const messageList = chatMain.querySelector(".message-list");
  const messageInput = chatMain.querySelector('.chat-composer [name="content"]');
  const editStatus = chatMain.querySelector("[data-question-edit-status]");

  if (!messageList || !messageInput || !editStatus) {
    return;
  }

  const resizeMessageInput = () => {
    if (messageInput instanceof HTMLTextAreaElement) {
      messageInput.style.removeProperty("width");
      messageInput.style.height = "auto";
      messageInput.style.height = `${messageInput.scrollHeight}px`;
    }
  };

  messageInput.addEventListener("input", resizeMessageInput);
  window.addEventListener("resize", resizeMessageInput);
  resizeMessageInput();

  messageList.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) {
      return;
    }

    const editButton = event.target.closest("[data-edit-question]");
    if (!editButton) {
      return;
    }

    const question = editButton
      .closest(".message-card")
      ?.querySelector(".message-text")?.textContent;
    if (!question) {
      return;
    }

    messageInput.value = question;
    resizeMessageInput();
    messageInput.focus();
    editStatus.textContent =
      "Вопрос загружен. Отредактируйте его и отправьте повторно.";
  });
});

document.querySelectorAll("[data-document-chat-form]").forEach((form) => {
  const documentCheckboxes = Array.from(form.elements).filter(
    (element) => element.name === "document_ids" && element.type === "checkbox",
  );
  const submitButton = form.querySelector('button[type="submit"]');
  const pendingStatus = form.querySelector("[data-pending-status]");
  const hasSelectedDocument = () =>
    documentCheckboxes.some((checkbox) => checkbox.checked);

  const updateSubmitButton = () => {
    if (submitButton) {
      submitButton.disabled = !hasSelectedDocument();
    }
    if (hasSelectedDocument() && pendingStatus) {
      pendingStatus.textContent = "";
      pendingStatus.hidden = true;
    }
  };

  documentCheckboxes.forEach((checkbox) => {
    checkbox.addEventListener("change", updateSubmitButton);
  });
  updateSubmitButton();

  form.addEventListener("submit", (event) => {
    if (!hasSelectedDocument()) {
      event.preventDefault();
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

    if (form.matches("[data-vector-upload-form]")) {
      const uploadStatus = document.querySelector("[data-vector-upload-status]");
      if (uploadStatus) {
        uploadStatus.hidden = false;
      }
    }

    pendingStatus.textContent = form.dataset.pendingMessage;
    pendingStatus.hidden = false;
    submitButton.textContent =
      form.dataset.pendingLabel || submitButton.textContent;
    submitButton.disabled = true;
    form.setAttribute("aria-busy", "true");

    const messageInput = form.querySelector('[name="content"]');
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

      const editButton = document.createElement("button");
      editButton.className = "message-edit-question";
      editButton.type = "button";
      editButton.dataset.editQuestion = "";
      editButton.textContent = "Редактировать вопрос";

      messageHeader.append(messageRole, messageTime);
      messageCard.append(messageHeader, messageText, editButton);
      messageList.append(messageCard);
      messageInput.value = "";
      messageInput.dispatchEvent(new Event("input", { bubbles: true }));
      messageList.scrollTop = messageList.scrollHeight;
    }
  });
});
