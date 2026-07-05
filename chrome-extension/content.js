/**
 * content.js
 *
 * Runs automatically on Gmail pages. Watches for an opened email,
 * injects a "Check Spam" button near it, and on click, sends the
 * email body text to our local FastAPI service for classification.
 *
 * NOTE: Gmail's HTML structure is not officially documented/stable,
 * so the selectors below (e.g. 'div.a3s') are based on Gmail's current
 * classic web layout and may break if Google changes their markup.
 * This is a known, honest limitation of unofficial Gmail integrations.
 */

const API_URL = "http://127.0.0.1:8000/predict";

// Gmail's email body container - the class 'a3s' is Gmail's (unofficial,
// long-standing) class name for the rendered email body content.
const EMAIL_BODY_SELECTOR = "div.a3s.aiL";

let lastCheckedElement = null;

function extractEmailText(emailBodyElement) {
    // innerText (not innerHTML) gives us the human-readable text,
    // stripping out HTML tags, styles, and hidden elements.
    return emailBodyElement.innerText.trim();
}

function createResultBanner(label, confidence) {
    const banner = document.createElement("div");
    banner.className = "spam-checker-banner " + (label === "Spam" ? "spam" : "ham");
    banner.innerText = label === "Spam"
        ? `⚠️ Likely SPAM (${(confidence * 100).toFixed(1)}% confidence)`
        : `✅ Looks like a legitimate email (${(confidence * 100).toFixed(1)}% confidence)`;
    return banner;
}

function createCheckButton(emailBodyElement) {
    const button = document.createElement("button");
    button.innerText = "🔍 Check Spam";
    button.className = "spam-checker-button";

    button.addEventListener("click", async () => {
        button.innerText = "Checking...";
        button.disabled = true;

        const text = extractEmailText(emailBodyElement);

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                throw new Error(`API returned status ${response.status}`);
            }

            const data = await response.json();

            // Remove any previous banner before adding a new one
            const existingBanner = document.querySelector(".spam-checker-banner");
            if (existingBanner) existingBanner.remove();

            const banner = createResultBanner(data.label, data.confidence);
            button.insertAdjacentElement("afterend", banner);

        } catch (error) {
            console.error("Spam Checker error:", error);
            alert(
                "Could not reach the spam-checking API.\n\n" +
                "Make sure your local API is running:\n" +
                "  cd src\n  uvicorn api:app --reload\n\n" +
                "Error: " + error.message
            );
        } finally {
            button.innerText = "🔍 Check Spam";
            button.disabled = false;
        }
    });

    return button;
}

function injectButtonIfNeeded() {
    const emailBodyElement = document.querySelector(EMAIL_BODY_SELECTOR);

    if (!emailBodyElement) return;

    // Avoid injecting duplicate buttons if this email is already being watched
    if (emailBodyElement === lastCheckedElement) return;

    // Remove any old button/banner from a previously opened email
    document.querySelectorAll(".spam-checker-button, .spam-checker-banner").forEach(el => el.remove());

    const button = createCheckButton(emailBodyElement);
    emailBodyElement.insertAdjacentElement("beforebegin", button);

    lastCheckedElement = emailBodyElement;
}

// Gmail loads content dynamically without full page reloads,
// so we use a MutationObserver to detect when new content (like an opened email) appears.
const observer = new MutationObserver(() => {
    injectButtonIfNeeded();
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Also try once immediately, in case an email is already open when the script loads
injectButtonIfNeeded();