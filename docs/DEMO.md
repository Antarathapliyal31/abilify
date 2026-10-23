# Recording the Streamlit demo GIF

The repo's README references `docs/demo.gif`. To record it on macOS:

1. Start the app:
   ```bash
   source venv/bin/activate
   streamlit run app.py
   ```
   This opens the UI at http://localhost:8501.

2. Record the screen:
   - Cmd+Shift+5 → "Record Selected Portion" → draw a tight box around the browser tab
   - Walk through: enter a PII-laden question (e.g. "Hi I'm John Doe (john@acme.com), can Abilify be taken with Prozac?"), click Ask, show the redacted-query expander, show the final answer, open the Debug expander
   - Stop recording — produces a `.mov`

3. Convert to GIF:
   ```bash
   # Install ffmpeg if needed: brew install ffmpeg
   ffmpeg -i ~/Desktop/recording.mov -vf "fps=10,scale=1000:-1:flags=lanczos" -loop 0 docs/demo.gif
   ```

4. Commit:
   ```bash
   git add docs/demo.gif
   git commit -m "Add demo GIF"
   ```

Alternative tools: Kap (free, built for this), ScreenToGif (Windows), peek (Linux).
