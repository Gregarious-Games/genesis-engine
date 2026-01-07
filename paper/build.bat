@echo off
echo Building Genesis Paper...
echo.

cd /d "%~dp0"

echo Step 1: First pdflatex pass...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto error

echo Step 2: BibTeX...
bibtex main 2>nul

echo Step 3: Second pdflatex pass...
pdflatex -interaction=nonstopmode main.tex

echo Step 4: Third pdflatex pass (final)...
pdflatex -interaction=nonstopmode main.tex
if errorlevel 1 goto error

echo.
echo ========================================
echo SUCCESS! PDF created: main.pdf
echo ========================================
echo.
start "" main.pdf
goto end

:error
echo.
echo ========================================
echo ERROR: pdflatex failed
echo Make sure MiKTeX is installed:
echo   winget install MiKTeX.MiKTeX
echo ========================================
echo.

:end
pause
