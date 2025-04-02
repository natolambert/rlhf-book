// Dark mode settings
const DARK_MODE_KEY = 'rlhfbook_darkMode'; // Specific key for this site

// Apply dark mode to all elements
function applyDarkMode(enable) {
  // Apply to root elements
  document.documentElement.classList[enable ? 'add' : 'remove']('dark-mode');
  document.querySelector('html').classList[enable ? 'add' : 'remove']('dark-mode');
  document.body.classList[enable ? 'add' : 'remove']('dark-mode');
  
  // Apply to specific containers for better styling
  document.querySelectorAll('header, #content, footer, nav, .section, .dropdown-content').forEach(element => {
    if (element) element.classList[enable ? 'add' : 'remove']('dark-mode');
  });
  
  // Apply to any iframes
  const iframes = document.querySelectorAll('iframe');
  iframes.forEach(iframe => {
    try {
      if (iframe.contentDocument) {
        iframe.contentDocument.documentElement.classList[enable ? 'add' : 'remove']('dark-mode');
        iframe.contentDocument.body.classList[enable ? 'add' : 'remove']('dark-mode');
      }
    } catch(e) {
      console.log('Could not access iframe content:', e);
    }
  });
  
  // Update button text
  const darkModeButton = document.getElementById('dark-mode-button');
  if (darkModeButton) {
    darkModeButton.classList.add('dark-mode-toggle'); // Ensure class is present
    darkModeButton.textContent = enable ? 'Light Mode' : 'Dark Mode';
  }
  
  console.log('Dark mode is now:', enable);
}

// Toggle dark mode and save preference
function toggleDarkMode() {
  console.log('Toggle dark mode button clicked');
  const isDarkMode = !document.documentElement.classList.contains('dark-mode');
  
  // Apply the new state
  applyDarkMode(isDarkMode);
  
  // Save preference with expiration (1 year)
  try {
    localStorage.setItem(DARK_MODE_KEY, isDarkMode.toString());
    
    // Also set a cookie for cross-subdomain support
    const expiryDate = new Date();
    expiryDate.setFullYear(expiryDate.getFullYear() + 1);
    document.cookie = `${DARK_MODE_KEY}=${isDarkMode}; expires=${expiryDate.toUTCString()}; path=/; domain=.rlhfbook.com; SameSite=Lax`;
  } catch (e) {
    console.error('Failed to save dark mode preference:', e);
  }
}

// Get dark mode preference from storage (localStorage or cookie)
function getDarkModePreference() {
  // Try localStorage first
  const localStoragePref = localStorage.getItem(DARK_MODE_KEY);
  if (localStoragePref !== null) {
    return localStoragePref === 'true';
  }
  
  // Fall back to cookies for cross-subdomain support
  const cookies = document.cookie.split(';');
  for (let cookie of cookies) {
    const [name, value] = cookie.trim().split('=');
    if (name === DARK_MODE_KEY) {
      return value === 'true';
    }
  }
  
  // Default to system preference if available
  if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return true;
  }
  
  return false;
}

// Apply dark mode immediately before DOM fully loads (prevents flash)
const isDarkMode = getDarkModePreference();
if (isDarkMode) {
  document.documentElement.classList.add('dark-mode');
  document.querySelector('html')?.classList.add('dark-mode');
}

// Initialize dark mode from preference on DOM load
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, checking for saved dark mode preference');
  
  // Add class to dark mode button
  const darkModeButton = document.getElementById('dark-mode-button');
  if (darkModeButton) {
    darkModeButton.classList.add('dark-mode-toggle');
    console.log('Added dark-mode-toggle class to button');
  }
  
  // Apply dark mode based on preference
  applyDarkMode(isDarkMode);
  
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
    
    /* Apply to all navigation elements */
    .dark-mode nav, nav.dark-mode,
    .dark-mode .navigation-dropdown, .navigation-dropdown.dark-mode,
    .dark-mode a.chapter-link, a.chapter-link.dark-mode,
    .dark-mode a.prev-chapter, a.prev-chapter.dark-mode,
    .dark-mode a.next-chapter, a.next-chapter.dark-mode {
      background-color: #222 !important;
      color: #e0e0e0 !important;
    }
    
    /* Target entire chapter navigation */
    .dark-mode #chapter-navigation, #chapter-navigation.dark-mode {
      background-color: #222 !important;
    }
  `;
  document.head.appendChild(style);
  
  // Listen for system theme changes
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
      // Only apply system preference if no manual preference has been set
      if (localStorage.getItem(DARK_MODE_KEY) === null) {
        applyDarkMode(e.matches);
      }
    });
  }
});