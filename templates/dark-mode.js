// Dark mode toggle functionality
function toggleDarkMode() {
  console.log('Toggle dark mode button clicked');
  // Apply to all root elements
  document.documentElement.classList.toggle('dark-mode');
  document.querySelector('html').classList.toggle('dark-mode');
  document.body.classList.toggle('dark-mode');
  
  // Apply to specific containers that might be causing scrolling issues
  document.querySelectorAll('header, #content, footer').forEach(element => {
    if (element) element.classList.toggle('dark-mode');
  });
  
  // Also apply to any iframes that might be on the page
  const iframes = document.querySelectorAll('iframe');
  iframes.forEach(iframe => {
    try {
      if (iframe.contentDocument) {
        iframe.contentDocument.documentElement.classList.toggle('dark-mode');
        iframe.contentDocument.body.classList.toggle('dark-mode');
      }
    } catch(e) {
      console.log('Could not access iframe content:', e);
    }
  });
  
  const isDarkMode = document.documentElement.classList.contains('dark-mode');
  console.log('Dark mode is now:', isDarkMode);
  localStorage.setItem('darkMode', isDarkMode);
  
  // Update button text based on current mode
  const darkModeButton = document.getElementById('dark-mode-button');
  if (darkModeButton) {
    darkModeButton.classList.add('dark-mode-toggle'); // Ensure class is present
    darkModeButton.textContent = isDarkMode ? 'Light Mode' : 'Dark Mode';
  }
}

// Initialize dark mode from localStorage
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, checking for saved dark mode preference');
  
  // Add class to dark mode button
  const darkModeButton = document.getElementById('dark-mode-button');
  if (darkModeButton) {
    darkModeButton.classList.add('dark-mode-toggle');
    console.log('Added dark-mode-toggle class to button');
  }
  
  const isDarkMode = localStorage.getItem('darkMode') === 'true';
  
  if (isDarkMode) {
    console.log('Applying saved dark mode preference');
    // Apply to all root elements
    document.documentElement.classList.add('dark-mode');
    document.querySelector('html').classList.add('dark-mode');
    document.body.classList.add('dark-mode');
    
    // Apply to specific containers that might be causing scrolling issues
    document.querySelectorAll('header, #content, footer').forEach(element => {
      if (element) element.classList.add('dark-mode');
    });
    
    // Also apply to any iframes that might be on the page
    const iframes = document.querySelectorAll('iframe');
    iframes.forEach(iframe => {
      try {
        if (iframe.contentDocument) {
          iframe.contentDocument.documentElement.classList.add('dark-mode');
          iframe.contentDocument.body.classList.add('dark-mode');
        }
      } catch(e) {
        console.log('Could not access iframe content:', e);
      }
    });
    
    // Update button text based on current mode
    const darkModeButton = document.getElementById('dark-mode-button');
    if (darkModeButton) {
      darkModeButton.textContent = 'Light Mode';
    }
  }
  
  // Force apply styles to ensure Pandoc default styles don't override
  const style = document.createElement('style');
  style.textContent = `
    html.dark-mode {
      background-color: #222 !important;
      color: #e0e0e0 !important;
    }
    
    body.dark-mode {
      background-color: #222 !important;
      color: #e0e0e0 !important;
    }
    
    .dark-mode a, a.dark-mode {
      color: #6a9ae6 !important;
    }
    
    .dark-mode a:visited, a.dark-mode:visited {
      color: #9980c4 !important;
    }
    
    .dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode h4, .dark-mode h5, .dark-mode h6,
    h1.dark-mode, h2.dark-mode, h3.dark-mode, h4.dark-mode, h5.dark-mode, h6.dark-mode {
      color: #f0f0f0 !important;
    }
    
    .dark-mode blockquote, blockquote.dark-mode {
      color: #a0a0a0 !important;
      border-left-color: #444 !important;
    }
    
    .dark-mode code, code.dark-mode {
      color: #ddd !important;
    }
    
    .dark-mode th, th.dark-mode {
      background-color: #333 !important;
      border-color: #555 !important;
    }
    
    .dark-mode td, td.dark-mode {
      border-color: #555 !important;
    }
    
    .dark-mode .dropdown-content, .dark-mode .dropdown-button,
    .dropdown-content.dark-mode, .dropdown-button.dark-mode {
      background-color: #222 !important;
      color: #e0e0e0 !important;
    }
    
    .dark-mode .section, .section.dark-mode {
      background-color: #222 !important;
      border-color: #555 !important;
    }
    
    .dark-mode .section a, .section.dark-mode a {
      color: #6a9ae6 !important;
    }
    
    /* Fix margins, header and any fixed elements */
    html.dark-mode header, body.dark-mode header {
      background-color: #222 !important;
    }
    
    html.dark-mode #content, body.dark-mode #content {
      background-color: #222 !important;
    }
    
    html.dark-mode footer, body.dark-mode footer {
      background-color: #222 !important;
    }
    
    /* Fix scrolling issues */
    html.dark-mode, body.dark-mode {
      overflow-y: auto !important;
    }
  `;
  document.head.appendChild(style);
});