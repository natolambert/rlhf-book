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

  // Lightweight HTML-to-markdown for clipboard
  function htmlToMarkdown(el) {
    var html = el.innerHTML;
    var tmp = document.createElement('div');
    tmp.innerHTML = html;
    return (function walk(node) {
      if (node.nodeType === 3) return node.textContent;
      if (node.nodeType !== 1) return '';
      var tag = node.tagName;
      var children = Array.prototype.map.call(node.childNodes, walk).join('');
      if (tag === 'STRONG' || tag === 'B') return '**' + children + '**';
      if (tag === 'EM' || tag === 'I') return '*' + children + '*';
      if (tag === 'CODE') return '`' + children + '`';
      if (tag === 'A') return '[' + children + '](' + (node.getAttribute('href') || '') + ')';
      if (tag === 'P') return children + '\n\n';
      if (tag === 'BR') return '\n';
      if (tag === 'DIV') return children;
      return children;
    })(tmp).replace(/\n{3,}/g, '\n\n').trim();
  }

  var COPY_SVG =
    '<svg class="copy-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
      '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
    '</svg>';
  var CHECK_SVG =
    '<svg class="check-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none">' +
      '<polyline points="20 6 9 17 4 12"></polyline>' +
    '</svg>';

  function addCopyButton(msgDiv) {
    var btn = document.createElement('button');
    btn.className = 'copy-conversation-button';
    btn.title = 'Copy message';
    btn.innerHTML = COPY_SVG + CHECK_SVG;
    btn.addEventListener('click', function () {
      var content = msgDiv.querySelector('.conversation-content');
      var text = htmlToMarkdown(content);
      navigator.clipboard.writeText(text).then(function () {
        btn.querySelector('.copy-icon').style.display = 'none';
        btn.querySelector('.check-icon').style.display = 'block';
        btn.classList.add('copied');
        setTimeout(function () {
          btn.querySelector('.copy-icon').style.display = 'block';
          btn.querySelector('.check-icon').style.display = 'none';
          btn.classList.remove('copied');
        }, 2000);
      });
    });
    msgDiv.appendChild(btn);
  }

  function wrapThinkingTokens(contentDiv) {
    // Detect <code>&lt;thinking&gt;</code> ... <code>&lt;/thinking&gt;</code> and wrap in collapsible <details>
    var html = contentDiv.innerHTML;
    var openTag = '<code>&lt;thinking&gt;</code>';
    var closeTag = '<code>&lt;/thinking&gt;</code>';
    var openIdx = html.indexOf(openTag);
    var closeIdx = html.indexOf(closeTag);
    if (openIdx === -1 || closeIdx === -1 || closeIdx <= openIdx) return;

    var before = html.substring(0, openIdx);
    var thinkingContent = html.substring(openIdx + openTag.length, closeIdx);
    var after = html.substring(closeIdx + closeTag.length);

    contentDiv.innerHTML =
      before +
      '<details class="thinking-block" open>' +
        '<summary>Thinking</summary>' +
        '<div class="thinking-content">' + thinkingContent + '</div>' +
      '</details>' +
      after;
  }

  function isConversationRole(label) {
    var lower = label.toLowerCase().trim();
    if (KNOWN_ROLES.indexOf(lower) !== -1) return true;
    if (lower.indexOf('assistant') !== -1) return true;
    if (lower.indexOf('verification') !== -1) return true;
    return false;
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
    var knownRoleCount = 0;

    for (var j = 0; j < paragraphs.length; j++) {
      var p = paragraphs[j];
      var parsed = parseRoleFromP(p);

      if (parsed && isConversationRole(parsed.label)) {
        knownRoleCount++;
        var roleClass = classifyRole(parsed.label);
        messages.push({
          role: roleClass,
          label: parsed.label,
          contentParts: [parsed.contentHTML]
        });
      } else {
        // Continuation paragraph (or non-conversation bold label) — append to previous message
        if (messages.length > 0) {
          messages[messages.length - 1].contentParts.push(p.innerHTML);
        }
      }
    }

    // Require at least one known conversation role
    if (knownRoleCount < 1) continue;

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
      wrapThinkingTokens(contentDiv);
      msgDiv.appendChild(contentDiv);
      addCopyButton(msgDiv);

      container.appendChild(msgDiv);
    }

    bq.parentNode.replaceChild(container, bq);
  }
});
