const copyButton = document.getElementById("copy-citation");
const bibtex = document.getElementById("bibtex");

if (copyButton && bibtex) {
  copyButton.addEventListener("click", async () => {
    const originalLabel = copyButton.textContent;

    try {
      await navigator.clipboard.writeText(bibtex.textContent.trim());
      copyButton.textContent = "Copied";
    } catch (error) {
      copyButton.textContent = "Select text";
      console.error("Unable to copy citation:", error);
    }

    window.setTimeout(() => {
      copyButton.textContent = originalLabel;
    }, 1600);
  });
}
