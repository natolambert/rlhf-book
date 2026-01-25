/**
 * Adds anchor links to all headings on the page.
 * When clicked, the anchor link copies the URL with fragment to the clipboard.
 */
document.addEventListener('DOMContentLoaded', function() {
  // Find all heading elements
  const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
  
  headings.forEach(function(heading) {
    // Skip headings without IDs
    if (!heading.id) return;
    
    // Create anchor element
    const anchor = document.createElement('a');
    anchor.classList.add('header-anchor');
    anchor.href = '#' + heading.id;
    anchor.textContent = '🔗';
    anchor.title = 'Copy link to this section';
    
    // Add click handler to copy the URL
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      
      // Create the full URL with fragment
      const url = window.location.href.split('#')[0] + '#' + heading.id;
      
      // Copy to clipboard
      navigator.clipboard.writeText(url).then(function() {
        // Visual feedback that link was copied
        const originalTitle = anchor.title;
        anchor.title = 'Link copied to clipboard!';
        
        // Reset title after a delay
        setTimeout(function() {
          anchor.title = originalTitle;
        }, 2000);
      }).catch(function(err) {
        console.error('Could not copy text: ', err);
      });
    });
    
    // Add the anchor to the heading
    heading.appendChild(anchor);
  });
});