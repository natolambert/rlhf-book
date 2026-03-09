/**
 * Adds a "Copy Markdown" button and "View Source" link below the chapter header.
 * Fetches the raw .md file on click and copies it to the clipboard.
 */
document.addEventListener('DOMContentLoaded', function() {
  var header = document.getElementById('title-block-header');
  if (!header) return;

  var path = window.location.pathname;
  var slug = path.substring(path.lastIndexOf('/') + 1).replace(/\.html$/, '');
  if (!slug) return;

  var mdFile = slug + '.md';
  var githubUrl = 'https://github.com/natolambert/rlhf-book/blob/main/book/chapters/' + mdFile;

  var COPY_SVG =
    '<svg class="copy-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
      '<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>' +
      '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>' +
    '</svg>';
  var CHECK_SVG =
    '<svg class="check-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:none">' +
      '<polyline points="20 6 9 17 4 12"></polyline>' +
    '</svg>';
  var GITHUB_SVG =
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">' +
      '<path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>' +
    '</svg>';

  // Build toolbar
  var toolbar = document.createElement('div');
  toolbar.className = 'chapter-toolbar';

  var copyBtn = document.createElement('button');
  copyBtn.className = 'copy-chapter-button';
  copyBtn.title = 'Copy raw chapter markdown to clipboard';
  copyBtn.innerHTML = COPY_SVG + CHECK_SVG + '<span class="copy-chapter-label">Copy Markdown</span>';

  var copyIcon = copyBtn.querySelector('.copy-icon');
  var checkIcon = copyBtn.querySelector('.check-icon');
  var label = copyBtn.querySelector('.copy-chapter-label');

  var cachedMarkdown = null;

  function showCopiedFeedback() {
    copyIcon.style.display = 'none';
    checkIcon.style.display = 'inline-block';
    copyBtn.classList.add('copied');

    setTimeout(function() {
      copyIcon.style.display = 'inline-block';
      checkIcon.style.display = 'none';
      copyBtn.classList.remove('copied');
    }, 2000);
  }

  function fetchMarkdown() {
    if (cachedMarkdown) return Promise.resolve(cachedMarkdown);
    return fetch(mdFile).then(function(res) {
      if (!res.ok) throw new Error('Failed to fetch ' + mdFile);
      return res.text();
    }).then(function(text) {
      cachedMarkdown = text;
      return text;
    });
  }

  copyBtn.addEventListener('click', function() {
    // ClipboardItem with a promise blob preserves the user-gesture context
    // in Safari even though fetch is async.
    var copyPromise;
    if (navigator.clipboard && navigator.clipboard.write && window.ClipboardItem) {
      var blobPromise = fetchMarkdown().then(function(text) {
        return new Blob([text], { type: 'text/plain' });
      });
      copyPromise = navigator.clipboard.write([
        new ClipboardItem({ 'text/plain': blobPromise })
      ]);
    } else {
      copyPromise = fetchMarkdown().then(function(text) {
        return navigator.clipboard.writeText(text);
      });
    }
    copyPromise.then(showCopiedFeedback).catch(function(err) {
      console.error('Could not copy chapter markdown: ', err);
    });
  });

  var sourceLink = document.createElement('a');
  sourceLink.className = 'chapter-github-link';
  sourceLink.href = githubUrl;
  sourceLink.target = '_blank';
  sourceLink.rel = 'noopener';
  sourceLink.title = 'View source on GitHub';
  sourceLink.innerHTML = GITHUB_SVG + '<span>View Source</span>';

  toolbar.appendChild(copyBtn);
  toolbar.appendChild(sourceLink);

  // Insert between table of contents and chapter content
  var content = document.getElementById('content');
  if (content) {
    content.parentNode.insertBefore(toolbar, content);
  } else {
    header.parentNode.insertBefore(toolbar, header.nextSibling);
  }
});
