document$.subscribe(function () {
  if (typeof mermaid === "undefined") {
    return;
  }
  mermaid.initialize({ startOnLoad: false });
  mermaid.run({ querySelector: ".mermaid" });
});
