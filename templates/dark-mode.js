// Dark mode toggle functionality
function toggleDarkMode() {
  console.log('Toggle dark mode button clicked');
  document.documentElement.classList.toggle('dark-mode');
  document.querySelector('html').classList.toggle('dark-mode');
  const isDarkMode = document.documentElement.classList.contains('dark-mode');
  console.log('Dark mode is now:', isDarkMode);
  localStorage.setItem('darkMode', isDarkMode);
}

// Initialize dark mode from localStorage
document.addEventListener('DOMContentLoaded', function() {
  console.log('DOM loaded, checking for saved dark mode preference');
  if (localStorage.getItem('darkMode') === 'true') {
    console.log('Applying saved dark mode preference');
    document.documentElement.classList.add('dark-mode');
    document.querySelector('html').classList.add('dark-mode');
  }
});