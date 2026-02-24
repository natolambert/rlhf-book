document.addEventListener("DOMContentLoaded", function () {
  var citations = document.querySelectorAll(".citation");
  var tooltip = null;

  function removeTooltip() {
    if (tooltip) {
      tooltip.remove();
      tooltip = null;
    }
  }

  citations.forEach(function (cite) {
    cite.addEventListener("mouseenter", function () {
      removeTooltip();

      var key = cite.getAttribute("data-cites");
      if (!key) return;

      // data-cites can have multiple keys separated by spaces; use the first
      var firstKey = key.split(/\s+/)[0];
      var entry = document.getElementById("ref-" + firstKey);
      if (!entry) return;

      var inline = entry.querySelector(".csl-right-inline");
      var text = inline ? inline.textContent : entry.textContent;
      if (!text || !text.trim()) return;

      tooltip = document.createElement("div");
      tooltip.className = "citation-tooltip";
      tooltip.textContent = text.trim();
      document.body.appendChild(tooltip);

      var rect = cite.getBoundingClientRect();
      var tooltipRect = tooltip.getBoundingClientRect();

      // Horizontal: center on the citation, but clamp to viewport
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
    });

    cite.addEventListener("mouseleave", function () {
      removeTooltip();
    });
  });
});
