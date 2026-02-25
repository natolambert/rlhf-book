/**
 * Adds a copy-to-clipboard button to all code blocks.
 * On hover the button fades in; on click it copies the code and shows a checkmark.
 */
document.addEventListener('DOMContentLoaded', function() {
  var blocks = document.querySelectorAll('pre');

  blocks.forEach(function(block) {
    var button = document.createElement('button');
    button.className = 'copy-code-button';
    button.title = 'Copy code';
    button.innerHTML =
      '<svg class="copy-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
        '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
        '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
      '</svg>' +
      '<svg class="check-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none">' +
        '<polyline points="20 6 9 17 4 12"></polyline>' +
      '</svg>';

    button.addEventListener('click', function() {
      var code = block.querySelector('code');
      var text = code ? code.textContent : block.textContent;

      navigator.clipboard.writeText(text).then(function() {
        button.querySelector('.copy-icon').style.display = 'none';
        button.querySelector('.check-icon').style.display = 'block';
        button.classList.add('copied');

        setTimeout(function() {
          button.querySelector('.copy-icon').style.display = 'block';
          button.querySelector('.check-icon').style.display = 'none';
          button.classList.remove('copied');
        }, 2000);
      }).catch(function(err) {
        console.error('Could not copy code: ', err);
      });
    });

    block.appendChild(button);
  });
});
