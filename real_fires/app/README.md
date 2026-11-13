# Fire Spread Explorer

One page for all study fires: pick a fire, then play or drag the timeline to see where it burned.

- `index.html`: the page. Open it directly in a browser (no server needed) or serve the folder.
- `data/<fire>.js`: one data file per fire, loaded when the fire is selected.

Both are generated; do not edit them by hand. The page template is
`src/firedata/templates/fire_page.html` and the builder is `firedata.web.build_app`.

```bash
# from real_fires/
python -m firedata.web bug_fire colwash biscar crosswhite --app   # rebuild app/
python scripts/test_app.py app/index.html                        # browser test (requirements-dev.txt)
```

A link straight to one fire: `index.html#colwash`.
