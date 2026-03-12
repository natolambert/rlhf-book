/**
 * Adds subtle copy-to-clipboard buttons to code blocks and display equations.
 */
document.addEventListener('DOMContentLoaded', function() {
  function createButton(className, title) {
    var button = document.createElement('button');
    button.className = className;
    button.type = 'button';
    button.title = title;
    button.setAttribute('aria-label', title);
    button.innerHTML =
      '<svg class="copy-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
      '</svg>' +
      '<svg class="check-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none">' +
        '<polyline points="20 6 9 17 4 12"></polyline>' +
      '</svg>';
    return button;
  }

  function setCopiedState(button) {
    button.querySelector('.copy-icon').style.display = 'none';
    button.querySelector('.check-icon').style.display = 'block';
    button.classList.add('copied');

    setTimeout(function() {
      button.querySelector('.copy-icon').style.display = 'block';
      button.querySelector('.check-icon').style.display = 'none';
      button.classList.remove('copied');
    }, 2000);
  }

  function copyText(text, button, label) {
    navigator.clipboard.writeText(text).then(function() {
      setCopiedState(button);
    }).catch(function(err) {
      console.error('Could not copy ' + label + ': ', err);
    });
  }

  function normalizeDisplayMath(text) {
    var trimmed = text.replace(/\u00a0/g, ' ').trim();
    if (!trimmed) {
      return '';
    }

    if (trimmed.slice(0, 2) === '\\[' && trimmed.slice(-2) === '\\]') {
      return trimmed.slice(2, -2).trim();
    }

    if (trimmed.slice(0, 2) === '$$' && trimmed.slice(-2) === '$$') {
      return trimmed.slice(2, -2).trim();
    }

    return trimmed;
  }

  var blocks = document.querySelectorAll('pre');
  blocks.forEach(function(block) {
    if (block.parentNode && block.parentNode.classList.contains('code-wrapper')) {
      return;
    }

    var wrapper = document.createElement('div');
    wrapper.className = 'code-wrapper';
    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(block);

    var button = createButton('copy-code-button', 'Copy code');
    button.addEventListener('click', function() {
      var code = block.querySelector('code');
      var text = code ? code.textContent : block.textContent;
      copyText(text, button, 'code');
    });

    wrapper.appendChild(button);
  });

  var mathBlocks = document.querySelectorAll('.math.display');
  mathBlocks.forEach(function(block) {
    if (!block.parentNode || block.parentNode.classList.contains('math-wrapper')) {
      return;
    }

    var latex = normalizeDisplayMath(block.textContent);
    if (!latex) {
      return;
    }

    var wrapper = document.createElement('span');
    wrapper.className = 'math-wrapper';
    block.parentNode.insertBefore(wrapper, block);
    wrapper.appendChild(block);

    var button = createButton('copy-math-button', 'Copy LaTeX');
    button.addEventListener('click', function(event) {
      event.preventDefault();
      event.stopPropagation();
      copyText(latex, button, 'LaTeX');
    });

    wrapper.appendChild(button);
  });
});
