-- kindle-math.lua
-- Pandoc Lua filter that renders Math nodes to PNG images for Kindle ePUB.
-- Kindle doesn't support MathML, so we convert each equation to a
-- tight-cropped PNG via pdflatex (standalone class) + pdftoppm.
--
-- Usage: pandoc --lua-filter book/scripts/kindle-math.lua ...
--
-- Requires: pdflatex, pdftoppm (from poppler: brew install poppler)
--           standalone.cls (sudo tlmgr install standalone)

local system = require("pandoc.system")
local sha1 = pandoc.utils.sha1

-- Cache directory for rendered math PNGs
local CACHE_DIR = "build/kindle-math-cache"
local DPI = "300"

-- LaTeX preamble for standalone document
local PREAMBLE = [[
\documentclass[border=2pt]{standalone}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\begin{document}
]]

local POSTAMBLE = [[
\end{document}
]]

-- Track whether we've shown a detailed error yet
local first_error_shown = false
local render_count = 0
local fail_count = 0

-- Run a shell command and return success boolean + combined output
local function run_cmd(cmd)
  local handle = io.popen(cmd .. " 2>&1")
  if not handle then
    return false, "(failed to execute command)"
  end
  local output = handle:read("*a")
  local ok = handle:close()
  -- io.popen/close: Lua 5.3+ returns true on success, Lua 5.1 returns 0
  local succeeded
  if type(ok) == "boolean" then
    succeeded = ok
  elseif type(ok) == "number" then
    succeeded = (ok == 0)
  else
    -- nil means failure in some Lua versions
    succeeded = false
  end
  return succeeded, output or ""
end

-- Check that a command exists on PATH
local function check_command(name, install_hint)
  local ok, _ = run_cmd("which " .. name)
  if not ok then
    io.stderr:write(string.format(
      "ERROR: %s not found.\n  %s\n", name, install_hint
    ))
    os.exit(1)
  end
end

-- One-time startup validation
local startup_checked = false

local function startup_checks()
  if startup_checked then return end
  startup_checked = true

  check_command("pdflatex",
    "Install basictex: brew install --cask basictex")
  check_command("pdftoppm",
    "Install poppler: brew install poppler (macOS) / apt install poppler-utils (Linux)")

  -- Verify standalone.cls is available by compiling a trivial document
  os.execute("mkdir -p " .. CACHE_DIR)
  local test_tex = CACHE_DIR .. "/_startup_test.tex"
  local fh = io.open(test_tex, "w")
  fh:write(PREAMBLE .. "$x$\n" .. POSTAMBLE)
  fh:close()

  local ok, output = run_cmd(string.format(
    "pdflatex -interaction=nonstopmode -output-directory=%s %s",
    CACHE_DIR, test_tex
  ))
  -- Clean up test files
  os.remove(CACHE_DIR .. "/_startup_test.tex")
  os.remove(CACHE_DIR .. "/_startup_test.pdf")
  os.remove(CACHE_DIR .. "/_startup_test.aux")
  os.remove(CACHE_DIR .. "/_startup_test.log")

  if not ok then
    io.stderr:write("\n=== kindle-math.lua: pdflatex startup test FAILED ===\n")
    -- Check for common missing package errors
    if output:match("standalone.cls") then
      io.stderr:write(
        "ERROR: standalone.cls not found.\n"
        .. "  Fix: sudo tlmgr install standalone\n\n"
      )
    else
      io.stderr:write("pdflatex output:\n" .. output:sub(1, 2000) .. "\n\n")
    end
    os.exit(1)
  end

  io.stderr:write("kindle-math.lua: startup check passed (pdflatex + pdftoppm OK)\n")
end

-- Render a LaTeX math string to PNG, return the path to the PNG file.
-- Returns nil on failure.
local function render_math_to_png(latex_src, math_type)
  startup_checks()

  -- Hash includes math type to distinguish display vs inline rendering
  local type_suffix = (math_type == "DisplayMath") and "D" or "I"
  local hash = sha1(latex_src .. type_suffix)
  local png_path = CACHE_DIR .. "/" .. hash .. ".png"

  -- Check cache
  local f = io.open(png_path, "r")
  if f then
    f:close()
    return png_path
  end

  -- Build LaTeX source
  -- Note: standalone class doesn't support \[...\] display math delimiters,
  -- so we use $\displaystyle ...$ for display math instead.
  local math_content
  if math_type == "DisplayMath" then
    math_content = "$\\displaystyle " .. latex_src .. "$"
  else
    math_content = "$" .. latex_src .. "$"
  end

  local tex_source = PREAMBLE .. math_content .. "\n" .. POSTAMBLE

  -- Use a temp directory for the LaTeX build
  local success, result_path = pcall(function()
    return system.with_temporary_directory("kindle-math", function(tmpdir)
      local tex_file = tmpdir .. "/math.tex"
      local pdf_file = tmpdir .. "/math.pdf"
      local png_base = tmpdir .. "/math"

      -- Write .tex file
      local fh = io.open(tex_file, "w")
      fh:write(tex_source)
      fh:close()

      -- Run pdflatex
      local latex_ok, latex_output = run_cmd(string.format(
        "pdflatex -interaction=nonstopmode -output-directory=%s %s",
        tmpdir, tex_file
      ))

      if not latex_ok then
        -- Show detailed error for the first failure only
        if not first_error_shown then
          first_error_shown = true
          io.stderr:write("\n=== First pdflatex failure detail ===\n")
          io.stderr:write("LaTeX source:\n" .. tex_source .. "\n")
          -- Try to read the .log file for more detail
          local log_file = tmpdir .. "/math.log"
          local lf = io.open(log_file, "r")
          if lf then
            local log_content = lf:read("*a")
            lf:close()
            -- Extract error lines
            local errors = {}
            for line in log_content:gmatch("[^\n]+") do
              if line:match("^!") or line:match("Error") or line:match("Fatal") then
                table.insert(errors, line)
              end
            end
            if #errors > 0 then
              io.stderr:write("LaTeX errors:\n")
              for _, e in ipairs(errors) do
                io.stderr:write("  " .. e .. "\n")
              end
            else
              -- Show last 30 lines of log
              local lines = {}
              for line in log_content:gmatch("[^\n]+") do
                table.insert(lines, line)
              end
              io.stderr:write("Last lines of log:\n")
              local start = math.max(1, #lines - 30)
              for idx = start, #lines do
                io.stderr:write("  " .. lines[idx] .. "\n")
              end
            end
          else
            io.stderr:write("pdflatex output:\n" .. latex_output:sub(1, 1000) .. "\n")
          end
          io.stderr:write("=== end detail ===\n\n")
        end
        return nil
      end

      -- Convert PDF to PNG with pdftoppm
      local ppm_ok, ppm_output = run_cmd(string.format(
        "pdftoppm -png -r %s -singlefile %s %s",
        DPI, pdf_file, png_base
      ))

      if not ppm_ok then
        if not first_error_shown then
          first_error_shown = true
          io.stderr:write("pdftoppm failed: " .. ppm_output:sub(1, 500) .. "\n")
        end
        return nil
      end

      -- pdftoppm -singlefile outputs <base>.png
      local rendered_png = png_base .. ".png"
      local check = io.open(rendered_png, "r")
      if not check then
        return nil
      end
      check:close()

      -- Copy to cache
      os.execute(string.format("cp %s %s", rendered_png, png_path))

      return png_path
    end)
  end)

  if success and result_path then
    render_count = render_count + 1
    return result_path
  else
    fail_count = fail_count + 1
    return nil
  end
end

-- The filter function for Math elements
function Math(el)
  local png_path = render_math_to_png(el.text, el.mathtype)

  if png_path then
    local css_class
    if el.mathtype == "DisplayMath" then
      css_class = "display-math"
    else
      css_class = "inline-math"
    end

    -- Use pandoc.Image for both display and inline math so pandoc's
    -- XHTML writer produces valid alt attributes. The LaTeX source
    -- serves as alt text for accessibility.
    local alt_text = pandoc.Str(el.text)
    return pandoc.Image({alt_text}, png_path, "", pandoc.Attr("", { css_class }))
  else
    -- Fallback: render as code so the build doesn't break
    io.stderr:write("WARNING: Failed to render math: " .. el.text:sub(1, 60) .. "...\n")
    return pandoc.Code(el.text)
  end
end

-- Print summary at end
function Pandoc(doc)
  -- Run the Math filter on the full document
  local result = doc:walk({ Math = Math })
  io.stderr:write(string.format(
    "\nkindle-math.lua: %d rendered, %d failed\n", render_count, fail_count
  ))
  return result
end
