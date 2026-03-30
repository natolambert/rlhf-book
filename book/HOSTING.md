# Hosting Options

This book is hosted on **Cloudflare Pages** with builds running in **GitHub Actions**.

## Current: Cloudflare Pages (via GitHub Actions Direct Upload)

The site is built by GitHub Actions and deployed to Cloudflare Pages. This hybrid approach gives us:
- **GitHub Actions**: macOS runner with LaTeX for PDF/EPUB generation
- **Cloudflare Pages**: static hosting, preview deployments, custom domains, and `_redirects` support

### How it works

1. Push to `main` triggers `.github/workflows/static.yml`
2. GitHub Actions builds HTML, PDF, and EPUB on macOS (with LaTeX)
3. The `cloudflare/wrangler-action` action uploads `build/html/` to Cloudflare Pages

### Setup Requirements

Add these secrets to GitHub (Settings > Secrets > Actions):
- `CLOUDFLARE_ACCOUNT_ID`: From the Cloudflare dashboard overview
- `CLOUDFLARE_API_TOKEN`: A token with `Account > Cloudflare Pages > Edit`

Create a Cloudflare Pages project named `rlhf-book` before the first deploy. Use the **Direct Upload** workflow rather than Cloudflare's Git build integration, because the book build depends on GitHub Actions' macOS + LaTeX setup.

### Adding Redirects

Chapter redirects live in `book/_redirects`. The Makefile copies that file into `build/html/`, and Cloudflare Pages parses it at deploy time.

Use the standard Pages `_redirects` format:

```text
/c/old-chapter-name /c/new-chapter-name 301
```

## Legacy: Netlify

Netlify is no longer used for this repo. If you need to revisit the old setup, inspect the Git history for previous versions of:
- `.github/workflows/static.yml`
- `netlify.toml`
- `book/HOSTING.md`

## Alternative: GitHub Pages

GitHub Pages remains a possible fallback, but it is a worse fit than Cloudflare Pages because chapter redirects depend on `_redirects` support.

## DNS Configuration

### For Cloudflare Pages
To add a custom domain:
1. Go to Workers & Pages > your Pages project > **Custom domains**
2. Select **Set up a domain**
3. Add `rlhfbook.com`

If you want to serve the apex domain (`rlhfbook.com`), the zone should be delegated to Cloudflare nameservers.
If you only want to serve a subdomain, you can point a `CNAME` at `<project>.pages.dev`.

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
