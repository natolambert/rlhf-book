/*
 * Light/dark theme toggle.
 *
 * The page <head> contains a tiny inline script that sets the initial
 * data-theme on <html> before first paint (from localStorage, falling back to
 * the OS prefers-color-scheme). This file wires up the toggle button: it flips
 * the theme, persists the explicit choice, and keeps the button's icon/label in
 * sync. Until the reader makes an explicit choice, the page keeps following the
 * OS setting (live, via the matchMedia change listener).
 */
(function () {
  var STORAGE_KEY = 'theme';

  function systemTheme() {
    return window.matchMedia &&
      window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light';
  }

  function storedTheme() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null;
    }
  }

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') || 'light';
  }

  function setTheme(theme, persist) {
    document.documentElement.setAttribute('data-theme', theme);
    if (persist) {
      try {
        localStorage.setItem(STORAGE_KEY, theme);
      } catch (e) {
        /* ignore storage failures (private mode, etc.) */
      }
    }
    updateButton(theme);
  }

  function updateButton(theme) {
    var btn = document.getElementById('theme-toggle');
    if (!btn) return;
    var isDark = theme === 'dark';
    var label = isDark ? 'Switch to light mode' : 'Switch to dark mode';
    btn.setAttribute('aria-pressed', String(isDark));
    btn.setAttribute('aria-label', label);
    btn.setAttribute('title', label);
  }

  document.addEventListener('DOMContentLoaded', function () {
    var btn = document.getElementById('theme-toggle');
    updateButton(currentTheme());
    if (btn) {
      btn.addEventListener('click', function () {
        setTheme(currentTheme() === 'dark' ? 'light' : 'dark', true);
      });
    }
  });

  // Follow the OS setting live, but only while the reader hasn't chosen one.
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onChange = function (e) {
      if (!storedTheme()) {
        setTheme(e.matches ? 'dark' : 'light', false);
      }
    };
    if (mq.addEventListener) {
      mq.addEventListener('change', onChange);
    } else if (mq.addListener) {
      mq.addListener(onChange);
    }
  }
})();
