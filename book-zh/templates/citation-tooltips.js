document.addEventListener("DOMContentLoaded", function () {
  var tooltip = null;

  function removeTooltip() {
    if (tooltip) {
      tooltip.remove();
      tooltip = null;
    }
  }

  function showTooltip(anchor, key) {
    removeTooltip();

    var entry = document.getElementById("ref-" + key);
    if (!entry) return;

    var inline = entry.querySelector(".csl-right-inline");
    var text = inline ? inline.textContent : entry.textContent;
    if (!text || !text.trim()) return;

    tooltip = document.createElement("div");
    tooltip.className = "citation-tooltip";
    tooltip.textContent = text.trim();
    document.body.appendChild(tooltip);

    var rect = anchor.getBoundingClientRect();
    var tooltipRect = tooltip.getBoundingClientRect();

    // Horizontal: center on the anchor, but clamp to viewport
    var left = rect.left + rect.width / 2 - tooltipRect.width / 2 + window.scrollX;
    left = Math.max(8 + window.scrollX, Math.min(left, window.innerWidth - tooltipRect.width - 8 + window.scrollX));

    // Vertical: prefer below, flip above if near bottom of viewport
    var spaceBelow = window.innerHeight - rect.bottom;
    var gap = 6;
    var top;
    if (spaceBelow < tooltipRect.height + gap + 20) {
      top = rect.top + window.scrollY - tooltipRect.height - gap;
      tooltip.classList.add("above");
    } else {
      top = rect.bottom + window.scrollY + gap;
    }

    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
    tooltip.style.opacity = "1";
  }

  // Attach to individual <a> links inside citations (handles grouped citations)
  var citationLinks = document.querySelectorAll(".citation a[href^='#ref-']");
  citationLinks.forEach(function (link) {
    var key = link.getAttribute("href").replace(/^#ref-/, "");

    link.addEventListener("mouseenter", function () {
      showTooltip(link, key);
    });

    link.addEventListener("mouseleave", function () {
      removeTooltip();
    });
  });

  // Fallback for citations without inner <a> links (e.g. plain text citations)
  var citations = document.querySelectorAll(".citation");
  citations.forEach(function (cite) {
    if (cite.querySelector("a[href^='#ref-']")) return;

    cite.addEventListener("mouseenter", function () {
      var key = cite.getAttribute("data-cites");
      if (!key) return;
      showTooltip(cite, key.split(/\s+/)[0]);
    });

    cite.addEventListener("mouseleave", function () {
      removeTooltip();
    });
  });
});
