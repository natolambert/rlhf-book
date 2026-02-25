/**
 * conversation.js — Transform blockquote-based conversations into chat-bubble UI.
 *
 * Detects blockquotes where paragraphs start with **RoleName**: and replaces
 * them with styled conversation bubbles (user/assistant/system).
 */
document.addEventListener('DOMContentLoaded', function () {
  var KNOWN_ROLES = ['user', 'human', 'prompt', 'assistant', 'response', 'ai', 'system', 'verification'];
  var USER_ROLES = ['user', 'human', 'prompt'];
  var SYSTEM_ROLES = ['system', 'verification'];
  // Everything else (including "assistant", "response", "ai", anything with "assistant") → assistant

  function classifyRole(label) {
    var lower = label.toLowerCase().trim();
    if (USER_ROLES.indexOf(lower) !== -1) return 'user';
    if (SYSTEM_ROLES.indexOf(lower) !== -1) return 'system';
    // Match "verification (unit tests)" or similar
    if (lower.indexOf('verification') !== -1) return 'system';
    // Match anything containing "assistant" (e.g. "Sycophantic assistant")
    if (lower.indexOf('assistant') !== -1) return 'assistant';
    if (lower === 'response' || lower === 'ai') return 'assistant';
    return 'assistant'; // default fallback
  }

  function hasTimestamp(label) {
    // Exclude podcast-style labels like "Lex Fridman (03:41:56)"
    return /\(\d{1,2}:\d{2}(:\d{2})?\)/.test(label);
  }

  function parseRoleFromP(p) {
    // Check if <p> starts with <strong>RoleName</strong>: (with optional whitespace)
    var firstChild = p.firstChild;
    if (!firstChild) return null;

    var strong = null;
    if (firstChild.nodeType === 1 && firstChild.tagName === 'STRONG') {
      strong = firstChild;
    } else if (firstChild.nodeType === 3 && firstChild.textContent.trim() === '' && firstChild.nextSibling && firstChild.nextSibling.tagName === 'STRONG') {
      strong = firstChild.nextSibling;
    }
    if (!strong) return null;

    var label = strong.textContent.trim();
    if (hasTimestamp(label)) return null;

    // Check for colon after the <strong>
    var afterStrong = strong.nextSibling;
    if (!afterStrong) return null;
    if (afterStrong.nodeType === 3) {
      var text = afterStrong.textContent;
      if (text.charAt(0) !== ':') return null;
    } else {
      return null;
    }

    // Build remaining content HTML (everything after "**Label**: ")
    var tempDiv = document.createElement('div');
    // Add text after the colon (skip ": ")
    var remainingText = afterStrong.textContent.replace(/^:\s*/, '');
    if (remainingText) {
      tempDiv.appendChild(document.createTextNode(remainingText));
    }
    var sibling = afterStrong.nextSibling;
    while (sibling) {
      tempDiv.appendChild(sibling.cloneNode(true));
      sibling = sibling.nextSibling;
    }

    return { label: label, contentHTML: tempDiv.innerHTML };
  }

  var blockquotes = document.querySelectorAll('blockquote');

  for (var i = 0; i < blockquotes.length; i++) {
    var bq = blockquotes[i];
    var paragraphs = bq.querySelectorAll(':scope > p');
    if (paragraphs.length === 0) continue;

    var messages = [];
    var roleLabelCount = 0;
    var hasKnownRole = false;

    for (var j = 0; j < paragraphs.length; j++) {
      var p = paragraphs[j];
      var parsed = parseRoleFromP(p);

      if (parsed) {
        roleLabelCount++;
        var roleClass = classifyRole(parsed.label);
        if (KNOWN_ROLES.indexOf(parsed.label.toLowerCase().trim()) !== -1 ||
            parsed.label.toLowerCase().indexOf('assistant') !== -1 ||
            parsed.label.toLowerCase().indexOf('verification') !== -1) {
          hasKnownRole = true;
        }
        messages.push({
          role: roleClass,
          label: parsed.label,
          contentParts: [parsed.contentHTML]
        });
      } else {
        // Continuation paragraph — append to previous message
        if (messages.length > 0) {
          messages[messages.length - 1].contentParts.push(p.innerHTML);
        }
      }
    }

    // Qualify: ≥2 role-labeled paragraphs, OR ≥1 with a known conversation role
    if (roleLabelCount < 2 && !(roleLabelCount >= 1 && hasKnownRole)) continue;

    // Build the conversation container
    var container = document.createElement('div');
    container.className = 'conversation';

    for (var k = 0; k < messages.length; k++) {
      var msg = messages[k];
      var msgDiv = document.createElement('div');
      msgDiv.className = 'conversation-message conversation-message--' + msg.role;

      var roleDiv = document.createElement('div');
      roleDiv.className = 'conversation-role';
      roleDiv.textContent = msg.label;
      msgDiv.appendChild(roleDiv);

      var contentDiv = document.createElement('div');
      contentDiv.className = 'conversation-content';
      for (var m = 0; m < msg.contentParts.length; m++) {
        var pEl = document.createElement('p');
        pEl.innerHTML = msg.contentParts[m];
        contentDiv.appendChild(pEl);
      }
      msgDiv.appendChild(contentDiv);

      container.appendChild(msgDiv);
    }

    bq.parentNode.replaceChild(container, bq);
  }
});
