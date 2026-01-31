# Hosting Options

This book can be hosted on either **Netlify** (current) or **GitHub Pages** (legacy).

## Current: Netlify

The site is deployed via Netlify, configured in `netlify.toml`.

**Advantages:**
- Native 301 redirects via `_redirects` file (better for SEO)
- Faster builds (caching)
- Deploy previews for PRs

**Build process:**
- Installs pandoc and pandoc-crossref
- Runs `make html && make files`
- Publishes `build/html/`

**Note:** PDF/EPUB are not built on Netlify (requires LaTeX). Build locally with `make` and commit the files, or download from GitHub releases.

### Adding Redirects

When reordering chapters, add redirects to `netlify.toml`:

```toml
[[redirects]]
  from = "/c/old-chapter-name"
  to = "/c/new-chapter-name"
  status = 301
```

Or create a `_redirects` file in `build/html/`:

```
/c/old-chapter  /c/new-chapter  301
```

## Alternative: GitHub Pages

The GitHub Actions workflow in `.github/workflows/static.yml` is commented out but preserved.

**Advantages:**
- Simpler setup (no external service)
- Builds PDF/EPUB (has LaTeX installed)
- Everything in one place

**Disadvantages:**
- No native 301 redirects (only HTML meta refresh)
- Slower builds

### Re-enabling GitHub Pages

1. Uncomment the workflow in `.github/workflows/static.yml`
2. Go to repo Settings → Pages → Source: **GitHub Actions**
3. Delete or rename `netlify.toml`
4. Push to `main` to trigger deployment
5. Update DNS if needed (point to GitHub's servers)

## DNS Configuration

### For Netlify
Netlify provides the DNS records when you add a custom domain in their dashboard.

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
