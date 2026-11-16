# Paper theme

The supplied DESIGN.md, tokens.json, variables.css and theme.reference.css are
kept as design references. The reference @theme syntax requires Tailwind and is
not loaded by Streamlit. Only variables.css and main.css are applied at runtime.
The invalid source swatch #ffda6 is overridden by the canonical #ffda6e.

ui_theme.py embeds local Inter font files and loads the CSS relative to its own
location. Inter is a permitted substitute in the reference and includes Cyrillic.
Fonts come from Google Fonts, licensed under the included SIL Open Font License.
Source: https://fonts.google.com/specimen/Inter

.streamlit/config.toml sets the matching native Streamlit light theme, including
canvas data tables. No frontend build, external font request or new dependency is
required. The adapter targets Streamlit 1.32's data-testid and data-baseweb
attributes instead of generated class names. Recheck selectors when upgrading
Streamlit.

The historical sidebar, resume uploader, matrix editor and result tabs remain.
Plotly changes are limited to typography, palette, surfaces and axis readability.
