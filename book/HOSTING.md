# Hosting Options

This book can be hosted on either **Netlify** (current) or **GitHub Pages** (legacy).

## Current: Netlify (via GitHub Actions)

The site is built by GitHub Actions and deployed to Netlify. This hybrid approach gives us:
- **GitHub Actions**: macOS runner with LaTeX for PDF/EPUB generation
- **Netlify**: Native 301 redirects for SEO

### How it works

1. Push to `main` triggers `.github/workflows/static.yml`
2. GitHub Actions builds HTML, PDF, and EPUB on macOS (with LaTeX)
3. The `nwtgck/actions-netlify` action deploys `build/html/` to Netlify

### Setup Requirements

Add these secrets to GitHub (Settings > Secrets > Actions):
- `NETLIFY_SITE_ID`: From Netlify site settings
- `NETLIFY_AUTH_TOKEN`: From Netlify user settings > Applications

### Adding Redirects

When reordering chapters, add redirects to `netlify.toml`:

```toml
[[redirects]]
  from = "/c/old-chapter-name"
  to = "/c/new-chapter-name"
  status = 301
```

## Alternative: GitHub Pages

To switch back to GitHub Pages (simpler but no 301 redirects):

1. Edit `.github/workflows/static.yml`:
   - Replace the "Deploy to Netlify" step with the GitHub Pages steps (see comments in file)
   - Add the required permissions block
2. Go to repo Settings > Pages > Source: **GitHub Actions**
3. Delete or rename `netlify.toml`
4. Update DNS to point to GitHub's servers

**Disadvantages of GitHub Pages:**
- No native 301 redirects (only HTML meta refresh, worse for SEO)

## DNS Configuration

### For Netlify
Netlify provides the DNS records when you add a custom domain in their dashboard.
Typically an `A` record or `ALIAS` to Netlify's load balancer.

### For GitHub Pages
Add these DNS records:
- `A` record: `185.199.108.153` (and .109, .110, .111)
- `CNAME` for `www`: `natolambert.github.io`

## Analytics

Analytics are handled by Plausible (privacy-friendly, JS-based). The script is included in all HTML templates:
- `book/templates/html.html` (index)
- `book/templates/chapter.html` (chapters)
- `book/templates/library.html` (library)

Plausible tracks page views and file downloads (PDF/EPUB clicks).
