"use strict";

document.querySelectorAll("form[data-pending-message]").forEach((form) => {
  form.addEventListener("submit", () => {
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
  });
});
