const test = require("node:test");
const assert = require("node:assert/strict");
const { formatTokens } = require("../static/js/text_builder.js");
test("datilologia e separadores são determinísticos", () => {
  assert.equal(
    formatTokens([
      { kind: "letter", text: "L" },
      { kind: "letter", text: "E" },
      { kind: "letter", text: "O" },
      { kind: "separator" },
      { kind: "separator" },
      { kind: "word", text: "Oi" },
    ]),
    "LEO Oi",
  );
});
test("não inventa verbos ou intenções", () => {
  assert.equal(
    formatTokens([
      { kind: "word", text: "Eu" },
      { kind: "word", text: "Casa" },
    ]),
    "Eu Casa",
  );
});
test("remoção opera em unidades reconhecidas", () => {
  const tokens = [
    { kind: "word", text: "Meu nome" },
    { kind: "letter", text: "A" },
  ];
  tokens.pop();
  assert.equal(formatTokens(tokens), "Meu nome");
});
