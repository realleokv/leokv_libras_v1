/* Estado tipado de transcrição; não interpreta intenções nem inventa verbos. */
(function (root) {
  "use strict";
  function formatTokens(tokens) {
    let text = "",
      letters = "";
    const flush = () => {
      if (letters) {
        text += (text && !text.endsWith(" ") ? " " : "") + letters;
        letters = "";
      }
    };
    for (const token of tokens) {
      if (token.kind === "letter") letters += token.text.toUpperCase();
      else if (token.kind === "separator") {
        flush();
        if (text && !text.endsWith(" ")) text += " ";
      } else if (token.kind === "word") {
        flush();
        text += (text && !text.endsWith(" ") ? " " : "") + token.text;
      }
    }
    flush();
    return text.replace(/\s+/g, " ").trim();
  }
  const api = { formatTokens };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LibrasText = api;
})(typeof window !== "undefined" ? window : globalThis);
