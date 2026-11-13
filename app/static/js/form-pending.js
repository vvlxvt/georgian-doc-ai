"use strict";

const chatScrollStorageKey = "chat-scroll-after-send";
const submittedChatPath = sessionStorage.getItem(chatScrollStorageKey);

if (submittedChatPath) {
  sessionStorage.removeItem(chatScrollStorageKey);
  if (submittedChatPath === window.location.pathname) {
    requestAnimationFrame(() => {
      document
        .querySelector(".message-list")
        ?.lastElementChild?.scrollIntoView({ block: "end" });
    });
  }
}

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

const uploadStepLabels = {
  upload: "Загрузка файла",
  extraction: "Извлечение текста",
  ocr: "Распознавание OCR",
  storage: "Сохранение данных",
  embeddings: "Создание embeddings",
  indexing: "Индексация в Qdrant",
};
const uploadStepStatusLabels = {
  pending: "Ожидает",
  processing: "Выполняется",
  completed: "Готово",
  failed: "Ошибка",
};

document.querySelectorAll("form[data-vector-upload-form]").forEach((form) => {
  const progress = document.querySelector("[data-upload-progress]");
  const badge = progress?.querySelector("[data-upload-badge]");
  const statusMessage = progress?.querySelector("[data-upload-message]");
  const submitButton = form.querySelector('button[type="submit"]');
  const originalButtonLabel = submitButton?.textContent;
  const uploadStatus = document.querySelector("[data-vector-upload-status]");

  if (!progress || !badge || !statusMessage || !submitButton) {
    return;
  }

  const setStepStatus = (step, status) => {
    const row = progress.querySelector(`[data-upload-step="${step}"]`);
    if (!row) {
      return;
    }

    row.hidden = false;
    row.dataset.status = status;
    const indicator = row.querySelector(".upload-step-indicator");
    const statusLabel = row.querySelector("[data-step-status]");
    if (indicator) {
      indicator.textContent =
        status === "completed" ? "✓" : status === "failed" ? "!" : "○";
    }
    if (statusLabel) {
      statusLabel.textContent = uploadStepStatusLabels[status] || status;
    }
  };

  const setOverallStatus = (status, message) => {
    badge.dataset.status = status;
    badge.textContent =
      status === "completed"
        ? "Готово"
        : status === "failed"
          ? "Ошибка"
          : status === "processing"
            ? "Обработка"
            : "Ожидает";
    statusMessage.textContent = message;
  };

  const resetProgress = () => {
    progress.querySelectorAll("[data-upload-step]").forEach((row) => {
      row.dataset.status = "pending";
      row.hidden = row.dataset.uploadStep === "ocr";
      row.querySelector(".upload-step-indicator").textContent = "○";
      row.querySelector("[data-step-status]").textContent =
        uploadStepStatusLabels.pending;
    });
  };

  const finishProgress = () => {
    progress.querySelectorAll('[data-upload-step][data-status="pending"]').forEach(
      (row) => {
        row.hidden = true;
      },
    );
    if (uploadStatus) {
      uploadStatus.hidden = true;
    }
  };

  const handleUploadEvent = (uploadEvent) => {
    if (uploadEvent.event === "progress") {
      setStepStatus(uploadEvent.step, uploadEvent.status);
      if (uploadEvent.status === "processing") {
        setOverallStatus(
          "processing",
          `${uploadStepLabels[uploadEvent.step] || "Обработка"}...`,
        );
      } else if (uploadEvent.status === "failed" && uploadEvent.message) {
        setOverallStatus("processing", uploadEvent.message);
      }
      return false;
    }

    if (uploadEvent.event === "completed") {
      progress.querySelectorAll('[data-upload-step][data-status="processing"]')
        .forEach((row) => setStepStatus(row.dataset.uploadStep, "completed"));
      finishProgress();
      setOverallStatus("completed", uploadEvent.message);
      return true;
    }

    if (uploadEvent.event === "failed") {
      setStepStatus(uploadEvent.step, "failed");
      setOverallStatus(
        "failed",
        `${uploadStepLabels[uploadEvent.step] || "Обработка"}: ${uploadEvent.message}`,
      );
      finishProgress();
      return true;
    }

    return false;
  };

  const readProgressStream = async (response) => {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Прогресс обработки недоступен.");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let terminalEventReceived = false;

    while (!terminalEventReceived) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

      let frameEnd = buffer.indexOf("\n\n");
      while (frameEnd !== -1) {
        const frame = buffer.slice(0, frameEnd);
        buffer = buffer.slice(frameEnd + 2);
        const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
        if (dataLine) {
          terminalEventReceived = handleUploadEvent(
            JSON.parse(dataLine.slice(6)),
          );
        }
        frameEnd = buffer.indexOf("\n\n");
      }

      if (done) {
        break;
      }
    }

    await reader.cancel();
    if (!terminalEventReceived) {
      throw new Error("Поток обновлений завершился до окончания обработки.");
    }
  };

  form.addEventListener("submit", async (event) => {
    if (event.defaultPrevented || form.getAttribute("aria-busy") === "true") {
      event.preventDefault();
      return;
    }

    event.preventDefault();
    resetProgress();
    setStepStatus("upload", "processing");
    setOverallStatus("processing", "Файл передаётся на сервер.");
    form.setAttribute("aria-busy", "true");
    submitButton.disabled = true;
    submitButton.textContent = form.dataset.pendingLabel || "Загрузка...";

    if (form.matches("[data-vector-upload-form]")) {
      if (uploadStatus) {
        uploadStatus.hidden = false;
      }
    }

    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "text/event-stream" },
      });

      if (response.headers.get("content-type")?.includes("text/event-stream")) {
        await readProgressStream(response);
      } else {
        const html = await response.text();
        const parsed = new DOMParser().parseFromString(html, "text/html");
        const message =
          parsed.querySelector(".alert-error")?.textContent.trim() ||
          "Не удалось принять файл. Проверьте PDF и повторите попытку.";
        setStepStatus("upload", "failed");
        setOverallStatus("failed", message);
        finishProgress();
      }
    } catch {
      setOverallStatus(
        "processing",
        "Соединение прервано. Обработка может продолжаться на сервере; проверьте список документов.",
      );
    } finally {
      form.removeAttribute("aria-busy");
      submitButton.disabled = false;
      submitButton.textContent = originalButtonLabel;
    }
  });
});

document
  .querySelectorAll(
    "form[data-pending-message]:not([data-vector-upload-form]), form.chat-composer",
  )
  .forEach((form) => {
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

      const messageInput = form.querySelector('[name="content"]');
      const pendingMessage =
        form.dataset.pendingMessage || messageInput?.dataset.pendingMessage;
      const pendingStatus = form.querySelector("[data-pending-status]");
      const submitButton = form.querySelector('button[type="submit"]');

      if (!pendingMessage || !pendingStatus || !submitButton) {
        return;
      }

      pendingStatus.textContent = pendingMessage;
      pendingStatus.hidden = false;
      submitButton.textContent =
        form.dataset.pendingLabel || submitButton.textContent;
      submitButton.disabled = true;
      form.setAttribute("aria-busy", "true");

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

        sessionStorage.setItem(chatScrollStorageKey, window.location.pathname);

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
        messageInput.placeholder = pendingMessage;
        messageInput.dispatchEvent(new Event("input", { bubbles: true }));
        messageList.scrollTop = messageList.scrollHeight;
      }
    });
  });
