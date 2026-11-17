# Paper theme

The supplied DESIGN.md, tokens.json, variables.css and theme.reference.css are
kept as design references. The reference @theme syntax requires Tailwind and is
not loaded by Streamlit. variables.css, main.css and glass.css are applied at runtime.
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

The requested iOS-inspired glass treatment is a separate layer in glass.css. It
retains the warm palette while intentionally adding translucent surfaces, soft
shadows, highlights, segmented selection groups and short entrance/press motions.
This supersedes the original reference's flat-only treatment. It preserves native
Streamlit controls, keyboard focus, disabled states and chart interactions.
Reduced-motion preferences disable transitions and entrance animations. Opaque
fallbacks cover browsers without backdrop-filter and reduced-transparency mode.

The native BaseWeb tab indicator is used as a sliding full-height pill: its width
and translation animate together. No DOM-injecting script is required. SVG trace,
bar and pie reveal animations leave Plotly data and interaction intact. The
analysis status panel reports actual model/file stages and completed-file counts;
it adds no artificial wait or fabricated timing. New results, glass matrix
previews and landing sections have entrance/scroll reveal motion, all disabled
under prefers-reduced-motion.
